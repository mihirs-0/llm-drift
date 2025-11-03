# scripts/train_gpt_e1.py
import argparse, math, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from scripts.utils_data import get_dataloader
from scripts.utils_tok import get_tokenizer

def default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def build_model(tokenizer_dir: Path):
    tok = GPT2TokenizerFast.from_pretrained(str(tokenizer_dir))
    cfg = GPT2Config(
        vocab_size=tok.vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_layer=6,
        n_head=8,
        n_embd=512,
        bos_token_id=None,
        eos_token_id=tok.eos_token_id,  # Required for GPT2Config
        pad_token_id=tok.pad_token_id,  # Required for GPT2Config       
    )
    model = GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(tok.vocab_size)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok

@torch.no_grad()
def eval_ppl(model, dl, device):
    model.eval()
    total_loss = 0.0
    total_tok  = 0
    for xb, mask, yb in dl:
        xb, mask, yb = xb.to(device), mask.to(device), yb.to(device)
        out = model(xb, attention_mask=mask, labels=yb)
        n_tok = (yb != -100).sum().item()
        if n_tok==0:
            continue
        # NOTE: out.loss is mean over non-ignored tokens
        total_loss += out.loss.item() * n_tok  # out.loss is mean over non-ignored tokens
        total_tok  += n_tok
    if total_tok==0:
        return float('inf'), float('inf')

    mean_loss = total_loss / total_tok
    ppl = math.exp(mean_loss)
    model.train()
    return ppl, mean_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", type=Path, required=True)
    ap.add_argument("--data_pt", type=Path, required=True)
    ap.add_argument("--dev_pt", type=Path, required=True)
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--device", type=str, default=default_device())

    args = ap.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1337)
    if args.device == "cuda":
        torch.cuda.manual_seed(1337)
    elif args.device == "mps":
        torch.mps.manual_seed(1337)

    model, tok = build_model(args.tokenizer_dir)
    model.to(args.device)

    try:
        model=torch.compile(model)
        print("Using torch.compile for training!")
    except Exception as e:
        print(f"torch.compile failed: {e}")
        print("Falling back to eager mode.")

    # dataloaders
    train_dl = get_dataloader(args.data_pt, batch_size=args.batch_size, shuffle=True)
    dev_dl   = get_dataloader(args.dev_pt,  batch_size=args.batch_size, shuffle=False)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    num_updates=args.max_steps 
    warmup = min(args.warmup_steps,max(1,num_updates//10))
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup, num_training_steps=num_updates
    )

    # --- training ---
    num_updates = args.max_steps
    micro_step = 0
    update_step = 0
    model.train()
    logs = []

    train_loss_sum = 0.0
    train_tok_sum  = 0

    done = False
    while not done:
        for xb, mask, yb in train_dl:
            xb, mask, yb = xb.to(args.device), mask.to(args.device), yb.to(args.device)

            out = model(xb, attention_mask=mask, labels=yb)

            # accumulate true train loss in dev units (token-mean)
            n_tok = (yb != -100).sum().item()
            if n_tok > 0:
                train_loss_sum += out.loss.item() * n_tok
                train_tok_sum  += n_tok

            # grad accumulation
            (out.loss / args.grad_accum).backward()
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # do not overstep the schedule budget
                if update_step < num_updates:
                    optim.step()
                    sched.step()
                optim.zero_grad(set_to_none=True)

                update_step += 1

                # periodic eval/log
                if update_step % args.eval_every == 0 or update_step == num_updates:
                    if train_tok_sum > 0:
                        train_mean = train_loss_sum / train_tok_sum
                        train_ppl  = math.exp(train_mean)
                    else:
                        train_mean = float("inf")
                        train_ppl  = float("inf")

                    dev_ppl, dev_loss = eval_ppl(model, dev_dl, args.device)
                    rec = {
                        "step": int(update_step),
                        "train_loss": float(train_mean),
                        "train_ppl": float(train_ppl),
                        "dev_loss": float(dev_loss),
                        "dev_ppl": float(dev_ppl),
                        "lr": float(sched.get_last_lr()[0]),
                    }
                    logs.append(rec)
                    print(rec, flush=True)

                    # reset running train stats after each log window
                    train_loss_sum = 0.0
                    train_tok_sum  = 0

                # hard-stop exactly at the budget
                if update_step >= num_updates:
                    done = True
                    break

    # final eval
    final_dev_ppl, final_dev_loss = eval_ppl(model, dev_dl, args.device)
    final={
        "final_dev_loss": final_dev_loss,
        "final_dev_ppl": final_dev_ppl,
        "steps": int(update_step),
    }
    print(final)

    # save artifacts
    (args.save_dir / "metrics.json").write_text(json.dumps({"log": logs, "final": final}, indent=2), encoding="utf-8")
    
    model.save_pretrained(str(args.save_dir))
    tok.save_pretrained(str(args.save_dir))

    print("Training complete! Model saved to:", args.save_dir)

if __name__ == "__main__":
    main()