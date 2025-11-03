# scripts/sample_e1.py
import argparse
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from scripts.utils_tok import encode_forward, encode_reversed, reverse_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--tokenizer_dir", type=Path, required=True)
    ap.add_argument("--direction", choices=["forward","reversed"], required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = GPT2TokenizerFast.from_pretrained(str(args.tokenizer_dir))
    model = GPT2LMHeadModel.from_pretrained(str(args.model_dir))
    model.config.pad_token_id = tok.pad_token_id
    model.to(args.device)
    model.eval()

    if args.direction == "forward":
        ids = encode_forward(tok, args.prompt)
    else:
        ids = encode_reversed(tok, args.prompt)

    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id
    )[0].tolist()

    if args.direction == "forward":
        text = tok.decode(out)
        print(text)
    else:
        # strip to the segment we generated and reverse back for readability
        # keep EOS anchored at end when reversing
        human_order = reverse_tokens(out, tok.eos_token_id)
        text = tok.decode(human_order)
        print(text)

if __name__ == "__main__":
    main()