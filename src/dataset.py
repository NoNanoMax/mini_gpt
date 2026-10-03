import json


def load_texts(path: str) -> list[str]:
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [d["text"] for d in data]
    with open(path, encoding="utf-8") as f:
        return [f.read()]


def make_batches(doc_ids: list[list[int]], batch_size: int, seq_len: int, eos_id: int,
                 shuffle: bool = False, seed: int = 42):
    import torch
    stream = []
    for doc in doc_ids:
        stream.extend(doc)
        stream.append(eos_id)
    chunks = [stream[i:i + seq_len] for i in range(0, len(stream) - seq_len, seq_len)]
    if shuffle:
        torch.manual_seed(seed)
        idx = torch.randperm(len(chunks)).tolist()
        chunks = [chunks[i] for i in idx]
    for i in range(0, len(chunks) - batch_size, batch_size):
        yield torch.tensor(chunks[i:i + batch_size], dtype=torch.long)
