# A2: Many-to-One Complexity Test (A->B vs B->A), random-init GPT-2 style model

import os, random, string
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# ---------- 1) Repro & device ----------
SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ---------- 2) Tokenizer ----------
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token  # define pad

# ---------- 3) Small random GPT-2 (no pretraining priors) ----------
cfg = GPT2Config(
    vocab_size=tok.vocab_size,
    n_layer=4,
    n_head=8,
    n_embd=512,
    n_positions=128,
    n_ctx=128,
    pad_token_id=tok.pad_token_id,
)

def get_fresh_model():
    return GPT2LMHeadModel(cfg).to(device)

# ---------- 4) Many-to-one data ----------
ALPH = list(string.ascii_lowercase + string.digits)

def rand_str(L: int) -> str:
    return "".join(random.choice(ALPH) for _ in range(L))

def gen_many_to_one_pairs(n_pairs: int, str_len: int, k_factor: int):
    """
    Generate pairs (A, B) such that k_factor different As map to the same B.
    So for every B, there are ~k_factor distinct As.

    This creates:
        A->B  : low conditional entropy (deterministic per sample)
        B->A  : high conditional entropy (same B appears with many As)
    """
    assert n_pairs % k_factor == 0, "n_pairs must be divisible by k_factor"
    n_targets = n_pairs // k_factor

    # Create a pool of unique B's
    targets = set()
    while len(targets) < n_targets:
        targets.add(rand_str(str_len))
    targets = list(targets)

    pairs = []
    for i in range(n_pairs):
        A = rand_str(str_len)
        B = targets[i % n_targets]  # cycle through limited B set
        pairs.append((A, B))

    random.shuffle(pairs)
    return pairs

# ---------- 5) Directional dataset (same as A1, with masking) ----------
# Template: "x: {INPUT} y: {TARGET}"

class DirectionalDataset(Dataset):
    def __init__(self, pairs, tokenizer, direction="A->B", max_len=64):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len
        assert direction in ("A->B", "B->A")
        self.direction = direction

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        A, B = self.pairs[i]

        if self.direction == "A->B":
            prompt_text = f"x: {A} y: "
            target_text = B
        else:  # "B->A"
            prompt_text = f"x: {B} y: "
            target_text = A

        # Tokenize prompt + target
        p_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        t_ids = self.tok(target_text, add_special_tokens=False).input_ids

        input_ids = p_ids + t_ids
        labels    = [-100] * len(p_ids) + t_ids  # predict only target tokens

        # Truncate politely (reserve 1 slot for EOS)
        if len(input_ids) >= self.max_len:
            input_ids = input_ids[: self.max_len - 1] + [self.tok.eos_token_id]
            labels    = labels[: self.max_len - 1]    + [-100]

        # Pad to fixed length
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tok.pad_token_id] * pad_len
            labels    = labels    + [-100] * pad_len

        attention_mask = [1 if tok_id != self.tok.pad_token_id else 0 for tok_id in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

# ---------- 6) Train / Eval helpers ----------

@torch.no_grad()
def eval_loss(model, dataset, batch_size=128):
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tot = 0.0
    steps = 0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        if batch["attention_mask"].dtype != torch.bool:
            batch["attention_mask"] = batch["attention_mask"].bool()
        out = model(**batch)
        tot += out.loss.item()
        steps += 1
    return tot / steps

def train_model(direction, train_pairs, val_pairs, epochs=3, lr=2e-4, bs=64):
    print(f"\n--- Train {direction} ---")
    model = get_fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_ds = DirectionalDataset(train_pairs, tok, direction=direction, max_len=64)
    val_ds   = DirectionalDataset(val_pairs,   tok, direction=direction, max_len=64)

    dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    tr_hist, va_hist = [], []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        steps = 0
        num_steps = len(dl)

        for step, batch in enumerate(dl):
            # progress ping every 100 steps
            if step % 100 == 0:
                print(f"[{direction}] epoch {ep}, step {step}/{num_steps}")

            batch = {k: v.to(device) for k, v in batch.items()}
            if batch["attention_mask"].dtype != torch.bool:
                batch["attention_mask"] = batch["attention_mask"].bool()

            out = model(**batch)
            loss = out.loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item()
            steps += 1

        tr = tot / steps
        va = eval_loss(model, val_ds)
        tr_hist.append(tr)
        va_hist.append(va)
        print(f"epoch {ep} | train {tr:.4f} | val {va:.4f}")

    return tr_hist, va_hist

# ---------- 7) Run A2 (Modified for Training Loss Focus) ----------
if __name__ == "__main__":
    N_PAIRS   = 10000   # total pairs
    STR_LEN   = 8       # string length
    K_FACTOR  = 5       # 5 different A's per B (entropy asymmetry)
    EPOCHS    = 3       
    LR        = 2e-4
    BATCH_SZ  = 64

    print(f"Generating {N_PAIRS} many-to-one pairs (K={K_FACTOR})...")
    pairs = gen_many_to_one_pairs(N_PAIRS, STR_LEN, K_FACTOR)
    
    # For random strings, we care about MEMORIZATION (Training Loss).
    # Validation on unseen random strings is impossible.
    # So we use the whole set for training to see 'how well can it fit?'
    train_pairs = pairs 
    # We create a dummy val set just so the code doesn't break, 
    # but we will ignore its values.
    val_pairs = pairs[:100] 

    # Forward: A -> B (low H)
    f_tr, _ = train_model(
        "A->B", train_pairs, val_pairs, epochs=EPOCHS, lr=LR, bs=BATCH_SZ
    )

    # Reverse: B -> A (high H)
    r_tr, _ = train_model(
        "B->A", train_pairs, val_pairs, epochs=EPOCHS, lr=LR, bs=BATCH_SZ
    )

    final_fwd = f_tr[-1]
    final_rev = r_tr[-1]
    gap = final_rev - final_fwd

    print("\n=== A2 Verdict (Many-to-One) ===")
    print(f"Final TRAIN Loss (A->B, Low H):  {final_fwd:.4f}")
    print(f"Final TRAIN Loss (B->A, High H): {final_rev:.4f}")
    print(f"Entropy Gap: {gap:.4f} nats")

    expected_entropy = 1.609 # ln(5)
    print(f"(Theoretical min entropy for 1-to-5 is ln(5) ≈ {expected_entropy:.3f})")

    if gap > 0.5:
        print("CONCLUSION: Hypothesis CONFIRMED.")
        print("The model resolved the low-entropy direction but hit the thermodynamic wall in reverse.")
    else:
        print("CONCLUSION: Unexpected result (gap is small).")