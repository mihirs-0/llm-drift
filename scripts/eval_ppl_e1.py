# scripts/eval_ppl_e1.py
import argparse, math, json, sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

try:
    from scripts.utils_data import get_dataloader
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from utils_data import get_dataloader


@torch.inference_mode()
def eval_ppl(model: GPT2LMHeadModel, dls, device: str) -> tuple[float, float]:
    """
    Token-weighted  mean loss across one or more DataLoaders.
    Returns (ppl, mean_loss, token_count)
    """

    model.eval()
    total_loss, total_tokens = 0.0, 0
    for dl in dls:
        for xb, mask, yb in dl:
            xb, mask, yb = xb.to(device), mask.to(device), yb.to(device)
            out = model(xb, attention_mask=mask, labels=yb)
            n_tok=(yb != -100).sum().item()
            if n_tok==0:
                continue
            total_loss += out.loss.item() * n_tok
            total_tokens += n_tok
        
    if total_tokens==0:
        raise RuntimeError("No tokens to evaluate on")
        
    mean_loss=total_loss/total_tokens

    mean_loss=float(max(min(mean_loss,50.0),-50.0))

    ppl=math.exp(mean_loss)

def default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    ap = argparse.ArgumentParser(description="Evaluate token-weighted perplexity of a pretokenized corpus")
    ap.add_argument("--model_dir", type=Path, required=True, help="Path to the pretrained model directory")
    ap.add_argument("--tokenizer_dir", type=Path, required=True, help="Path to the frozen tokenizer directory")
    ap.add_argument("--data_pt", type=Path, required=True, help="Path to the pre-tokenized corpus file")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", type=str, default=default_device())
    ap.add_argument("--out_json", type=Path, default=Path("eval_ppl.json"),help="Path to the output JSON file")
    args = ap.parse_args()
    
    # For MPS precision:
    if args.device=="mps":
        torch.set_float32_matmul_precision("medium")

    tok = GPT2TokenizerFast.from_pretrained(str(args.tokenizer_dir))
    if tok.pad_token_id is None or tok.eos_token_id is None:
        sys.exit("Tokenizer missing pad/eos. Run freeze_tokenizer.py first.")

    model = GPT2LMHeadModel.from_pretrained(str(args.model_dir))
    model.config.pad_token_id = tok.pad_token_id
    model.to(args.device)

    # Quick compat sanity check:
    if model.get_input_embeddings().weight.size(0)!=tok.vocab_size:
        sys.exit(f"Model vocab size mismatch with tokenizer: {model.get_input_embeddings().weight.size(0)} != {tok.vocab_size}"
        f"vs tokenizer.vocab_size={tok.vocab_size}")

    # Build dataloader:
    dls=[]
    for pt in args.data_pt:
        if not pt.exists():
            sys.exit(f"Missing pre-tokenized file: {pt}")
        dl= get_dataloader(pt, batch_size=args.batch_size, shuffle=False)
        
        dl.num_workers=0
        dl.pin_memory=False
        dls.append(dl)
    
    ppl,mean_loss,token_count=eval_ppl(model,dls,args.device)
    result={
        "files": [str(pt) for pt in args.data_pt],
        "tokens":int(n_tok),
        "mean_loss":float(mean_loss),
        "ppl":float(ppl),
        "device":args.device,
        "model_dir":str(args.model_dir),
        "tokenizer_dir":str(args.tokenizer_dir),
        "batch_size":args.batch_size,
        "out_json":str(args.out_json),
    }

    print(json.dumps(result, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Results saved to: {args.out_json}")
    else:
        print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()