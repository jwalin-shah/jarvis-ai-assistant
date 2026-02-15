import os

from mlx_lm import load

MODELS = [
    "mlx-community/LFM2-350M-4bit",
    "models/lfm2-1.2b-extract-mlx-4bit"
]

def check_vocabs():
    os.environ["HF_HUB_OFFLINE"] = "1"
    for m_path in MODELS:
        print(f"Loading {m_path}...")
        try:
            _, tokenizer = load(m_path)
            print(f"  Vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            print(f"  Failed to load: {e}")

if __name__ == "__main__":
    check_vocabs()
