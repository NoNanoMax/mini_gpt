import argparse
import json

import torch

from model import Config, GPT
from tokenizer import Tokenizer
from weights_io import load_safetensors


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    with open(f"{args.model}/config.json") as f:
        cfg = Config.from_dict(json.load(f))
    tokenizer = Tokenizer.load(f"{args.model}/tokenizer.json")
    model = GPT(cfg)
    state = load_safetensors(f"{args.model}/weights.safetensors")
    model.load_state_dict(state, strict=False)
    model.eval()

    ids = ([tokenizer.bos_id] if args.prompt == "" else tokenizer.encode(args.prompt))
    input_ids = torch.tensor([ids])
    out = model.generate(input_ids, args.n, temperature=args.temp, top_k=args.top_k,
                         eos_id=tokenizer.eos_id)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
