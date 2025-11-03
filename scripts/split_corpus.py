# scripts/split_corpus.py
import gzip, random, sys, os
from pathlib import Path

SRC = Path(sys.argv[1])  # path to corpus.txt.gz
OUT = Path(sys.argv[2])  # output dir
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1337

OUT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)

# stream read -> memory shuffle (1M lines will fit fine)
lines = []
with gzip.open(SRC, "rt", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s:
            lines.append(s)

random.shuffle(lines)

n = len(lines)
n_train = int(0.8 * n)
n_dev   = int(0.1 * n)
n_test  = n - n_train - n_dev

def dump(name, data):
    (OUT / name).write_text("\n".join(data), encoding="utf-8")

dump("train.txt", lines[:n_train])
dump("dev.txt",   lines[n_train:n_train+n_dev])
dump("test.txt",  lines[n_train+n_dev:])
print({"total": n, "train": n_train, "dev": n_dev, "test": n_test})