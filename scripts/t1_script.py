# T1: Many-to-One Memorization Test (Training Loss Only)
# Goal: Show that when H(B|A)=0 and H(A|B)=ln(k), SGD fits one direction and hits entropy floor in the reverse.

import os, random, string
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# ----------------------- 0) DEVICE & SEED -----------------------
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

# ----------------------- 1) TOKENIZER -----------------------
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

# ----------------------- 2) FRESH GPT-2 MODEL -----------------------
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

# ----------------------- 3) DATA GENERATION (Many-to-One) -----------------------
ALPH = list(string.ascii_lowercase + string.digits)

def rand_str(L):
    return "".join(random.choice(ALPH) for _ in range(L))

def gen_many_to_one_pairs(n_pairs, str_len, k_factor):
    assert n_pairs % k_factor == 0
    n_targets = n_pairs // k_factor

    targets = set()
    while len(targets) < n_targets:
        targets.add(rand_str(str_len))
    targets = list(targets)

    pairs = []
    for i in range(n_pairs):
        A = rand_str(str_len)
        B = targets[i % n_targets]
        pairs.append((A, B))

    random.shuffle(pairs)
    return pairs

# ----------------------- 4) DIRECTIONAL DATASET -----------------------
class DirectionalDataset(Dataset):
    def __init__(self, pairs, tokenizer, direction="A->B", max_len=64):
        assert direction in ("A->B", "B->A")
        self.pairs = pairs
        self.tok = tokenizer
        self.direction = direction
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        A, B = self.pairs[i]

        if self.direction == "A->B":
            prompt = f"x: {A} y: "
            target = B
        else:
            prompt = f"x: {B} y: "
            target = A

        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        t_ids = self.tok(target, add_special_tokens=False).input_ids

        input_ids  = p_ids + t_ids
        labels     = [-100]*len(p_ids) + t_ids

        # truncate
        if len(input_ids) >= self.max_len:
            input_ids = input_ids[:self.max_len-1] + [self.tok.eos_token_id]
            labels    = labels[:self.max_len-1]    + [-100]

        pad = self.max_len - len(input_ids)
        if pad > 0:
            input_ids += [self.tok.pad_token_id]*pad
            labels    += [-100]*pad

        attn = [1 if x!=self.tok.pad_token_id else 0 for x in input_ids]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

# ----------------------- 5) TRAINING -----------------------
@torch.no_grad()
def eval_train_loss(model, dataset, batch_size=128):
    """Evaluate TRAIN loss: same data, no shuffling effects."""
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tot, steps = 0.0, 0
    for batch in dl:
        batch = {k:v.to(device) for k,v in batch.items()}
        if batch["attention_mask"].dtype != torch.bool:
            batch["attention_mask"] = batch["attention_mask"].bool()
        out = model(**batch)
        tot += out.loss.item()
        steps += 1
    return tot/steps

def train_model(direction, train_pairs, epochs=3, lr=2e-4, bs=64):
    print(f"\n--- TRAINING {direction} ---")
    model = get_fresh_model()
    ds = DirectionalDataset(train_pairs, tok, direction=direction)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = []

    for ep in range(1, epochs+1):
        model.train()
        tot = steps = 0
        for step, batch in enumerate(dl):
            if step % 100 == 0:
                print(f"[{direction}] epoch {ep}, step {step}/{len(dl)}")

            batch = {k:v.to(device) for k,v in batch.items()}
            if batch["attention_mask"].dtype != torch.bool:
                batch["attention_mask"] = batch["attention_mask"].bool()

            out = model(**batch)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item()
            steps += 1

        tr_loss = eval_train_loss(model, ds)
        hist.append(tr_loss)
        print(f"Epoch {ep} | TRAIN loss = {tr_loss:.4f}")

    return hist

# ----------------------- 6) RUN T1 -----------------------
if __name__ == "__main__":
    N = 10000
    STR_LEN = 8
    K = 5
    EPOCHS = 7

    print(f"Generating many-to-one pairs: N={N}, K={K}")
    pairs = gen_many_to_one_pairs(N, STR_LEN, K)

    # Training set = entire set (memorization test)
    train_pairs = pairs

    f_tr = train_model("A->B", train_pairs, epochs=EPOCHS)
    r_tr = train_model("B->A", train_pairs, epochs=EPOCHS)

    # ----------------------- FINAL VERDICT -----------------------
    print("\n=== T1 VERDICT ===")
    print(f"Final TRAIN (A->B): {f_tr[-1]:.4f}")
    print(f"Final TRAIN (B->A): {r_tr[-1]:.4f}")

    gap = r_tr[-1] - f_tr[-1]
    expected = float(torch.log(torch.tensor(K)))

    print(f"Gap: {gap:.4f}")
    print(f"Expected entropy ln({K}) = {expected:.4f}")
    print("\nIf gap ≈ ln(K): CONDITIONAL ENTROPY BARRIER CONFIRMED.")