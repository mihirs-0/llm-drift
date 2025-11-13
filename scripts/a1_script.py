# A1: Bijection Parity Test (A->B vs B->A), random-init GPT-2 style model
import os, random, string
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# ---------- 1) Repro & device ----------
SEED = 1337
random.seed(SEED); torch.manual_seed(SEED)
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
    n_layer=4, n_head=8, n_embd=512,
    n_positions=128, n_ctx=128,
    pad_token_id=tok.pad_token_id
)
def get_fresh_model():
    return GPT2LMHeadModel(cfg).to(device)

# ---------- 4) Bijection data ----------
ALPH = list(string.ascii_lowercase + string.digits)  # 36 chars
def rand_str(L): return "".join(random.choice(ALPH) for _ in range(L))
def make_cipher(chars):
    shuf = chars[:]
    random.shuffle(shuf)
    fwd = dict(zip(chars, shuf))
    return fwd, {v:k for k,v in fwd.items()}

def apply_map(s, m): return "".join(m[c] for c in s)

def gen_bijection_pairs(n_pairs, str_len):
    # 1-to-1 mapping (true bijection)
    fwd_map, _ = make_cipher(ALPH)
    pairs = []
    seen = set()
    while len(pairs) < n_pairs:
        A = rand_str(str_len)
        if A in seen: 
            continue
        seen.add(A)
        B = apply_map(A, fwd_map)
        pairs.append((A, B))
    random.shuffle(pairs)
    return pairs

# ---------- 5) Directional dataset (proper loss masking) ----------
# Symmetric template: "x: {INPUT} y: {TARGET}"
class DirectionalDataset(Dataset):
    def __init__(self, pairs, tokenizer, direction="A->B", max_len=64):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len
        self.direction = direction
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        A, B = self.pairs[i]
        if self.direction == "A->B":
            prompt_text = f"x: {A} y: "
            target_text = B
        else:  # "B->A"
            prompt_text = f"x: {B} y: "
            target_text = A

        p_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
        t_ids = self.tok(target_text, add_special_tokens=False).input_ids
        input_ids = p_ids + t_ids
        labels    = [-100]*len(p_ids) + t_ids  # predict only target

        # truncate politely, keep shapes aligned
        if len(input_ids) >= self.max_len:
            input_ids = input_ids[:self.max_len-1] + [self.tok.eos_token_id]
            labels    = labels[:self.max_len-1]    + [-100]
        
        # pad to max_len for batching
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tok.pad_token_id] * pad_len
            labels    = labels + [-100] * pad_len
        
        attn = [1 if id != self.tok.pad_token_id else 0 for id in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

# ---------- 6) Train / Eval ----------
@torch.no_grad()
def eval_loss(model, dataset, batch_size=128):
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tot = steps = 0
    for batch in dl:
        batch = {k:v.to(device) for k,v in batch.items()}
        # Convert attention_mask to bool if needed
        if batch["attention_mask"].dtype != torch.bool:
            batch["attention_mask"] = batch["attention_mask"].bool()
        tot += model(**batch).loss.item()
        steps += 1
    return tot/steps

def train_model(direction, train_pairs, val_pairs, epochs=3, lr=2e-4, bs=64):
    print(f"\n--- Train {direction} ---")
    model = get_fresh_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_ds = DirectionalDataset(train_pairs, tok, direction=direction, max_len=64)
    val_ds   = DirectionalDataset(val_pairs,   tok, direction=direction, max_len=64)
    dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

    tr_hist, va_hist = [], []
    for ep in range(1, epochs+1):
        model.train(); tot=steps=0
        for batch in dl:
            batch = {k:v.to(device) for k,v in batch.items()}
            # Convert attention_mask to bool if needed
            if batch["attention_mask"].dtype != torch.bool:
                batch["attention_mask"] = batch["attention_mask"].bool()
            out = model(**batch); loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); steps += 1
        tr = tot/steps
        va = eval_loss(model, val_ds)
        tr_hist.append(tr); va_hist.append(va)
        print(f"epoch {ep} | train {tr:.4f} | val {va:.4f}")
    return tr_hist, va_hist

# ---------- 7) Run A1 ----------
N = 60000       # total pairs
L = 8           # string length
pairs = gen_bijection_pairs(N, L)
split = int(0.9*len(pairs))
train_pairs, val_pairs = pairs[:split], pairs[split:]

f_tr, f_va = train_model("A->B", train_pairs, val_pairs, epochs=3, lr=2e-4, bs=64)
r_tr, r_va = train_model("B->A", train_pairs, val_pairs, epochs=3, lr=2e-4, bs=64)

gap = r_va[-1] - f_va[-1]
print("\n=== A1 Verdict (Bijection) ===")
print(f"Final val NLL (A->B): {f_va[-1]:.4f}")
print(f"Final val NLL (B->A): {r_va[-1]:.4f}")
print(f"Directional Gap (R-F): {gap:.4f} nats/token")
assert abs(gap) < 0.1, "Non-trivial gap on a bijection suggests a confound."
print("PASS: No meaningful directional advantage on a true bijection.")