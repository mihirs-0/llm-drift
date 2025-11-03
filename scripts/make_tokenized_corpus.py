# scripts/make_tokenized_corpus.py
import argparse, sys, json
from pathlib import Path
from typing import List, Iterable
import torch
from tqdm import tqdm

# local utils
try:
    from scripts.utils_tok import encode_forward, encode_reversed
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.utils_tok import encode_forward, encode_reversed
from transformers import GPT2TokenizerFast

def read_lines(p: Path, max_lines: int | None) -> Iterable[str]:
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            s = line.strip()
            if s:
                yield s

def encode_lines(lines: Iterable[str], direction: str, tok: GPT2TokenizerFast) -> Iterable[List[int]]:
    enc = encode_forward if direction == "forward" else encode_reversed
    eos = tok.eos_token_id
    for s in lines:
        ids = enc(tok, s)
        # guard: must end with EOS
        if not ids or ids[-1] != eos:
            continue
        yield ids

def concat_and_chunk(id_iter: Iterable[List[int]], seq_len: int, stride: int, pad_id: int, drop_last: bool):
    """
    Stream tokens into a rolling buffer; emit fixed-length windows.
    No giant all_token_ids list. Your Mac thanks you.
    """
    buffer: List[int] = []
    total_in = 0
    chunks: list[list[int]] = []

    def flush_full_windows():
        nonlocal buffer
        out = []
        start = 0
        while start + seq_len <= len(buffer):
            out.append(buffer[start:start+seq_len])
            start += stride
        buffer = buffer[start:]
        return out

    for ids in id_iter:
        buffer.extend(ids)
        total_in += len(ids)
        new_chunks = flush_full_windows()
        if new_chunks:
            chunks.extend(new_chunks)

    # tail
    if buffer:
        if not drop_last:
            x = buffer[:seq_len]
            if len(x) < seq_len:
                x = x + [pad_id] * (seq_len - len(x))
            chunks.append(x)
        # else drop the remainder

    return chunks, total_in

def save_pt(path: Path, chunks: list[list[int]], pad_id: int, eos_id: int, seq_len: int, stride: int, direction: str, total_in: int):
    if not chunks:
        raise RuntimeError(f"No chunks for {path}")
    input_ids = torch.tensor(chunks, dtype=torch.long)
    attention_mask = (input_ids != pad_id)
    labels = input_ids.clone()
    labels[~attention_mask] = -100

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "meta": {
            "direction": direction,
            "seq_len": seq_len,
            "stride": stride,
            "num_chunks": int(input_ids.size(0)),
            "total_tokens_in": int(total_in),
            "total_tokens_used": int(attention_mask.sum().item()),
            "pad_token_id": int(pad_id),
            "eos_token_id": int(eos_id),
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)

def process_split(split: str, split_dir: Path, out_dir: Path, tok: GPT2TokenizerFast, direction: str, seq_len: int, stride: int, max_lines: int | None, drop_last: bool):
    src = split_dir / f"{split}.txt"
    if not src.exists():
        # Make 'test' split optional
        if split == 'test':
            print(f"\n[{split}] optional split not found at {src}. Skipping.")
            return None
        raise FileNotFoundError(f"Missing {src}")
    
    print(f"\n[{split}] reading → encoding({direction}) → chunking(seq={seq_len}, stride={stride})")
    lines = read_lines(src, max_lines)
    ids_iter = encode_lines(lines, direction, tok)
    chunks, total_in = concat_and_chunk(ids_iter, seq_len, stride, tok.pad_token_id, drop_last)
    
    if not chunks:
        print(f"[{split}] No chunks were created. Skipping save.")
        return None
        
    out = out_dir / f"{split}.pt" # Simplified name, as dir is already {direction}
    save_pt(out, chunks, tok.pad_token_id, tok.eos_token_id, seq_len, stride, direction, total_in)
    print(f"[{split}] chunks: {len(chunks)} | tokens_in: {total_in} -> saved: {out}")
    return {"file": str(out), "num_chunks": len(chunks), "total_tokens_in": total_in}

def main():
    ap = argparse.ArgumentParser(description="Pre-tokenize and chunk corpus into .pt")
    ap.add_argument("--split_dir", type=Path, required=True, help="Dir with train.txt/dev.txt/test.txt")
    ap.add_argument("--tokenizer_dir", type=Path, default=Path("tokenizer"))
    ap.add_argument("--out_dir", type=Path, required=True, help="Base output dir. Script will create {out_dir}/{direction}/")
    ap.add_argument("--direction", choices=["forward","reversed"], required=True)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=None, help="Default: seq_len (no overlap)")
    ap.add_argument("--max_lines", type=int, default=None, help="Cap lines per split for smoke tests")
    ap.add_argument("--drop_last", action="store_true", help="Drop trailing partial chunk")
    args = ap.parse_args()

    if args.stride is None:
        args.stride = args.seq_len
    if args.seq_len <= 8 or not (1 <= args.stride <= args.seq_len):
        sys.exit("Bad args: require seq_len > 8 and 1 ≤ stride ≤ seq_len")

    # use the frozen tokenizer you saved
    if not args.tokenizer_dir.exists():
        sys.exit(f"Tokenizer dir not found: {args.tokenizer_dir}")
    tok = GPT2TokenizerFast.from_pretrained(str(args.tokenizer_dir))
    if tok.pad_token_id is None or tok.eos_token_id is None:
        sys.exit("Tokenizer missing pad/eos. Run freeze_tokenizer.py first.")
    print(f"Tokenizer ok. eos={tok.eos_token_id} pad={tok.pad_token_id}")

    # Create the specific output directory for this direction
    direction_out_dir = args.out_dir / args.direction
    direction_out_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    for split in ["train","dev","test"]:
        try:
            split_stats = process_split(
                split, args.split_dir, direction_out_dir, tok, 
                args.direction, args.seq_len, args.stride, 
                args.max_lines, args.drop_last
            )
            if split_stats:
                stats[split] = split_stats
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as e:
            print(f"Error processing {split}: {e}", file=sys.stderr)
            sys.exit(1)

    # Save stats *inside* the direction-specific folder
    (direction_out_dir / f"stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nAll splits done.")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()