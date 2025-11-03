# scripts/utils_data.py
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

def _is_cuda():
    return torch.cuda.is_available()

def _is_mps():
    return getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

def _default_num_workers():
    # On macOS (spawn) with big in-memory TensorDataset, multiple workers duplicate memory.
    # On CPU/MPS: 0. On CUDA: you *can* try >0, but keep it modest.
    if _is_cuda():
        return 2
    return 0

def _default_pin_memory():
    # Only useful for CUDA
    return _is_cuda()

def get_dataloader(
    pt_file: Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    """
    Load a pre-tokenized/chunked .pt file and return a DataLoader.
    Expected .pt keys:
      - input_ids: LongTensor [N, seq_len]
      - labels:    LongTensor [N, seq_len]
      - attention_mask: BoolTensor [N, seq_len]
    """
    if not pt_file.exists():
        raise FileNotFoundError(
            f"Missing pre-tokenized file: {pt_file}\n"
            "Run 'scripts/make_tokenized_corpus.py' first."
        )

    # Load once on CPU; tensors are CPU by construction from make_tokenized_corpus
    try:
        data = torch.load(str(pt_file))
    except Exception as e:
        raise RuntimeError(f"Error loading data from {pt_file}: {e}")

    required = {"input_ids", "labels", "attention_mask"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"{pt_file} missing keys: {sorted(missing)}; found: {list(data.keys())}")

    dataset = TensorDataset(
        data["input_ids"],        # [N, L] long
        data["attention_mask"],   # [N, L] bool
        data["labels"],           # [N, L] long with -100 mask
    )

    # Sensible defaults per device
    if num_workers is None:
        num_workers = _default_num_workers()
    if pin_memory is None:
        pin_memory = _default_pin_memory()

    # Don’t keep workers alive if there aren’t any. Shocker.
    persistent_workers = bool(num_workers > 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

if __name__ == "__main__":
    print("--- 🧪 utils_data smoke test ---")
    test_dir = Path("data_e1_tok_test")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test.pt"

    N, L = 10, 8
    torch.save(
        {
            "input_ids": torch.randint(0, 1000, (N, L), dtype=torch.long),
            "labels": torch.randint(0, 1000, (N, L), dtype=torch.long),
            "attention_mask": torch.ones(N, L, dtype=torch.bool),
        },
        test_file,
    )
    dl = get_dataloader(test_file, batch_size=2, shuffle=False)
    x, m, y = next(iter(dl))
    print("shapes:", x.shape, m.shape, y.shape)
    test_file.unlink(missing_ok=True)
    test_dir.rmdir()
    print("✅ ok")