# MiniGPT

Написание, обучение и инференс (cpp + python) языковых моделей.

- **Python (PyTorch)** — обучение и инференс: BPE-токенизатор, GPT (RoPE, RMSNorm, SwiGLU, GQA)
- **C++17** — инференс-движок с KV-кэшем 

## Обучение

```bash
# датасет: data/corpus.txt (или .json списка {"text": ...})
python src/train.py --model configs/tiny.json --data configs/data.json
# продолжение:
python src/train.py --model configs/tiny.json --data configs/data.json --resume
```

Результат в `experiments/<name>/`:
- `weights.safetensors`, `config.json`, `tokenizer.json` — всё, что нужно C++-инференсу
- `checkpoint.pt` — для `--resume`

## Генерация

Python (проверка pipeline):

```bash
python src/generate.py --model experiments/tiny --prompt "Привет" --n 128
```

C++:

```bash
cmake -B cpp/build -S cpp && cmake --build cpp/build
./cpp/build/microgpt --model experiments/tiny --prompt "Привет" --n 128 --temp 0.8 --top-k 40
```

## Как это устроено

- Токены: BPE, обучается на корпусе; спецтокены `<bos> <eos> <pad>`; пробел приклеен к концу слова
- Модель: pre-norm transformer, RoPE, RMSNorm, SwiGLU, сращенный QKV, GQA
- Safetensors: `[len:8][JSON-заголовок][байты тензоров]`, dtypes F32/F16
