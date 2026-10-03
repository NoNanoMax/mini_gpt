import io
import json
import struct

import numpy as np
import torch

DTYPE_MAP = {
    torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int32: "I32", torch.int64: "I64",
}
TORCH_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
                "I32": torch.int32, "I64": torch.int64}


def save_safetensors(path: str, tensors: dict[str, torch.Tensor], dtype: torch.dtype = torch.float32):
    header, chunks, offset = {}, [], 0
    for name, t in tensors.items():
        t = t.detach().contiguous().cpu().to(dtype)
        nbytes = t.numel() * t.element_size()
        header[name] = {"dtype": DTYPE_MAP[t.dtype], "shape": list(t.shape),
                        "data_offsets": [offset, offset + nbytes]}
        chunks.append(t.numpy().tobytes())
        offset += nbytes
    pad = (-offset) % 8
    blob = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for c in chunks:
            f.write(c)
        f.write(b"\x00" * pad)


def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        blob = f.read()
    (hlen,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8:8 + hlen])
    base = 8 + hlen
    out = {}
    for name, meta in header.items():
        arr = np.frombuffer(
            blob[base + meta["data_offsets"][0]:base + meta["data_offsets"][1]],
            dtype=TORCH_DTYPES[meta["dtype"]].__str__().split('.')[-1])
        out[name] = torch.from_numpy(arr).view(meta["shape"]).clone()
    return out
