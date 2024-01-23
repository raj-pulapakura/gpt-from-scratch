# -----------------------------
# IMPORTS

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------
# LOAD DATASET

def load_dataset():
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { s:i for i, s in enumerate(chars) }
    itos = { i:s for s, i in stoi.items() }
    encode = lambda s: [stoi[c] for c in s] # encodes a string
    decode = lambda e: "".join([itos[i] for i in e]) # decodes an encoding
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, encode, decode

# -----------------------------
# HYPERPARAMETERS

train_data, val_data, VOCAB_SIZE, encode, decode = load_dataset()
BATCH_SIZE = 32
CONTEXT_LEN = 8
N_EMBED = 24
HEAD_SIZE = 24
LR = 1e-3
TRAIN_ITERS = 10000
EVAL_ITERS = 200

# -----------------------------
# HELPER FUNCTIONS

def get_batch(split):
    data = train_data if split == "train" else val_data
    # get "batch_size" number of random indices
    ixs = torch.randint(low=0, high=len(data)-CONTEXT_LEN, size=(BATCH_SIZE,))
    # get inputs
    x = torch.stack([data[i:i+CONTEXT_LEN] for i in ixs])
    # get labels
    y = torch.stack([data[i+1:i+CONTEXT_LEN+1] for i in ixs])
    return x, y

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------
# ARCHITECTURE

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.key = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x) # (B, T, head_size)
        # compute scaled dot production self-attention
        unmasked_attention = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        masked_attention = unmasked_attention.masked_fill(self.tril[:T, :T]==0, -torch.inf) # (B, T, T)
        self_attention = F.softmax(masked_attention, dim=-1) # (B, T, T)
        # apply attention
        v = self.value(x) # (B, T, head_size)
        out = self_attention @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out # (B, T, head_size)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # convert integers to embeddings
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        # add positional embeddings to retain positional information
        self.position_embedding_table = nn.Embedding(CONTEXT_LEN, N_EMBED)
        self.sa_head = Head()
        self.lm_head = nn.Linear(HEAD_SIZE, VOCAB_SIZE)

    def forward(self, x, y=None):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C), positional embeddings are broadcasted across batch
        x = self.sa_head(x) # (B, T, head_size)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, x, max_new_tokens):
        # x: (B, T)
        for _ in range(max_new_tokens):
            # crop x to the last context_len tokens
            x_crop = x[:, -CONTEXT_LEN:] # (B, T)
            # get predictions
            logits, _ = self(x_crop) # logits: (B, C, T)
            # for each prediction, get last timestep prediction
            logits = logits[:, -1, :] # (B, C)
            # calculate probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            # append sampled index to the running sequence
            x = torch.cat((x, idx_next), dim=1) # (B, T+1)
        return x

if __name__ == "__main__":
    # ===== instantiate model =====
    model = BigramLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ===== sample generation before training =====
    inputs = torch.zeros((1, 1), dtype=torch.long)
    generation = model.generate(inputs, max_new_tokens=100)
    print("Sample generation before training")
    print("---------------------------------")
    print(decode(generation.squeeze().tolist()))

    # ===== training =====
    print("\nTraining")
    print("--------")
    for steps in range(1, TRAIN_ITERS+1):
        # get sample batch
        xb, yb = get_batch("train")
        # calculate loss
        logits, loss = model(xb, yb)
        # set gradients to zero
        optimizer.zero_grad(set_to_none=True)
        # calculate gradients
        loss.backward()
        # update parameters
        optimizer.step()
        # track stats
        if steps % 1000 == 0 or steps == 1:
            print(f"{steps}/{TRAIN_ITERS}  {loss.item()}")

    # ===== estimate loss =====
    print("\nEstimated loss")
    print("--------------")
    print(estimate_loss(model))

    # ===== sample generation after training =====
    inputs = torch.zeros((1, 1), dtype=torch.long)
    generation = model.generate(inputs, max_new_tokens=200)
    print("\nSample generation after training")
    print("--------------------------------")
    print("".join(decode(generation.squeeze().tolist())))