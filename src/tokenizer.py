from collections import Counter


BOS, EOS, PAD = "<bos>", "<eos>", "<pad>"
SPECIALS = [BOS, EOS, PAD]


class Tokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self.vocab = vocab
        self.merges = merges
        self.id_to_piece = {i: p for p, i in vocab.items()}
        self.merge_rank = {(a, b): i for i, (a, b) in enumerate(merges)}
        self.vocab_size = len(vocab)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "Tokenizer":
        chars: dict[str, int] = {}
        for s in SPECIALS:
            chars[s] = len(chars)
        for c in text:
            if c not in chars:
                chars[c] = len(chars)

        words = [tuple(w) for w in text.split()]
        counts = Counter(words)

        merges: list[tuple[str, str]] = []
        while len(chars) < vocab_size:
            pairs = Counter()
            for word, n in counts.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += n
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            piece = best[0] + best[1]
            if piece in chars:
                break
            merges.append(best)
            chars[piece] = len(chars)
            new_counts = Counter()
            for word, n in counts.items():
                new_word = list(word)
                i = 0
                while i < len(new_word) - 1:
                    if (new_word[i], new_word[i + 1]) == best:
                        new_word[i] = piece
                        del new_word[i + 1]
                    else:
                        i += 1
                new_counts[tuple(new_word)] += n
            counts = new_counts
        return cls(chars, merges)

    def encode(self, text: str) -> list[int]:
        words = text.split(" ")
        ids = []
        for j, word in enumerate(words):
            if not word:
                continue
            symbols = list(word + (" " if j < len(words) - 1 else ""))
            i = 0
            while i < len(symbols) - 1:
                pair = (symbols[i], symbols[i + 1])
                if pair in self.merge_rank:
                    symbols[i] = symbols[i] + symbols[i + 1]
                    del symbols[i + 1]
                else:
                    i += 1
            ids.extend(self.vocab[s] for s in symbols)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_piece[i] for i in ids if i in self.id_to_piece)

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": [list(m) for m in self.merges]}, f,
                      ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["vocab"], [tuple(m) for m in data["merges"]])

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS]
