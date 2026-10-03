import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from model import Config, GPT
from tokenizer import Tokenizer
from dataset import load_texts, make_batches
from weights_io import save_safetensors


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.model) as f:
        mcfg = json.load(f)
    with open(args.data) as f:
        dcfg = json.load(f)

    out_dir = args.out or f"experiments/{mcfg['name']}"
    os.makedirs(out_dir, exist_ok=True)
    set_seed(dcfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Токенизация датасета...")
    t0 = time.time()
    tok_path = dcfg.get("tokenizer")
    tokenizer = Tokenizer.load(tok_path) if tok_path and os.path.exists(tok_path) else None
    if tokenizer is None:
        texts = load_texts(dcfg["data"])
        tokenizer = Tokenizer.train("".join(texts), mcfg["vocab_size"])
        tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
        dcfg["tokenizer"] = os.path.join(out_dir, "tokenizer.json")
        with open(args.data, "w") as f:
            json.dump(dcfg, f, indent=2)
        print(f"  токенизатор обучен: {tokenizer.vocab_size} токенов, {time.time() - t0:.1f}s")
    else:
        texts = load_texts(dcfg["data"])
        print(f"  токенизатор загружен, {time.time() - t0:.1f}s")

    doc_ids = [tokenizer.encode(t) for t in texts]
    total = sum(len(d) for d in doc_ids)
    print(f"Датасет: {len(doc_ids)} доков, {total} токенов")

    cfg = Config.from_dict({**mcfg, "vocab_size": tokenizer.vocab_size})
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: {n_params / 1e6:.1f}M параметров, device={device}")

    step, state = 0, None
    if args.resume and os.path.exists(os.path.join(out_dir, "checkpoint.pt")):
        state = torch.load(os.path.join(out_dir, "checkpoint.pt"), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        step = state["step"]
        print(f"Resume с шага {step}")

    opt = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    if state:
        opt.load_state_dict(state["opt"])

    batches = make_batches(doc_ids, dcfg["batch_size"], dcfg["seq_len"], tokenizer.eos_id,
                           shuffle=dcfg.get("shuffle", True))
    it = iter(batches)
    use_amp = device.type == "cuda"
    scaler_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else \
        torch.amp.autocast(device_type="cpu", enabled=False)

    running = 0.0
    t1 = time.time()
    for step in range(step, mcfg["steps"]):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(make_batches(doc_ids, dcfg["batch_size"], dcfg["seq_len"], tokenizer.eos_id))
            batch = next(it)
        x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)

        with scaler_ctx:
            logits = model(x).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), mcfg.get("grad_clip", 1.0))
        opt.param_groups[0]["lr"] = cosine_lr(step, mcfg["steps"], mcfg["lr"], mcfg.get("warmup", 100))
        opt.step()

        running += loss.item()
        if step % mcfg.get("log_every", 20) == 0:
            ppl = math.exp(min(running / mcfg.get("log_every", 20), 20))
            dt = time.time() - t1
            print(f"step {step:6d}  loss {running / mcfg.get('log_every', 20):.4f}  "
                  f"ppl {ppl:8.1f}  lr {opt.param_groups[0]['lr']:.2e}  {dt:.1f}s", flush=True)
            running, t1 = 0.0, time.time()

        if (step + 1) % mcfg.get("save_every", 500) == 0 or step + 1 == mcfg["steps"]:
            path = os.path.join(out_dir, "weights.safetensors")
            save_safetensors(path, model.state_dict_named(),
                             dtype=torch.float16 if mcfg.get("export_dtype") == "f16" else torch.float32)
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(cfg.to_dict(), f, indent=2)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1},
                       os.path.join(out_dir, "checkpoint.pt"))
            print(f"  сохранено: {path} ({os.path.getsize(path) / 1e6:.0f} MB)")

    print("Готово.")


if __name__ == "__main__":
    main()
