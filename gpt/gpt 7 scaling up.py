# Don't even try training this on CPU, go to colab and use a GPU, trust me

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
BATCH_SIZE = 64 # number of samples per batch
CONTEXT_LEN = 256 # number of previous chars used to predict next char
N_EMBED = 384 # token embedding dimension
NUM_HEADS = 6 # number of heads in multi-head attention
HEAD_SIZE = 384 # embedding dimension outputted by multi-head attention
N_BLOCKS = 6 # number of transformer blocks
LR = 3e-4 # learning rate
DROPOUT = 0.2 # dropout rate
TRAIN_ITERS = 10000 # number of training batches to process
EVAL_ITERS = 200 # number of val batches used to estimate loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    x, y = x.to(device), y.to(device)
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

class AttentionHead(nn.Module):
    """one head of self attention"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_LEN, CONTEXT_LEN)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, head_size)
        k = self.key(x) # (B, T, head_size)
        # compute scaled dot production self-attention
        unmasked_attention = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        masked_attention = unmasked_attention.masked_fill(self.tril[:T, :T]==0, -torch.inf) # (B, T, T)
        self_attention = F.softmax(masked_attention, dim=-1) # (B, T, T)
        self_attention = self.dropout(self_attention)
        # apply attention
        v = self.value(x) # (B, T, head_size)
        out = self_attention @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention"""

    def __init__(self, input_dim):
        super().__init__()
        dim_per_head = HEAD_SIZE // NUM_HEADS 
        self.heads = nn.ModuleList([AttentionHead(input_dim, dim_per_head) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(HEAD_SIZE, input_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer follwed by a non-linearity"""

    def __init__(self, inout_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inout_dim, inout_dim*4),
            nn.ReLU(),
            nn.Linear(inout_dim*4, inout_dim),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, input_dim):
        super().__init__()
        self.sa = MultiHeadAttention(input_dim)
        self.ffwd = FeedForward(input_dim)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        """x + is the residual connections"""
        x = x + self.sa(self.ln1(x)) # (B, T, input_dim)
        x = x + self.ffwd(self.ln2(x)) # (B, T, input_dim)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding_table = nn.Embedding(CONTEXT_LEN, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED) for _ in range(N_BLOCKS)])
        self.ln_f = nn.LayerNorm(N_EMBED) # final layer norm
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE) # logits

    def forward(self, x, y=None):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x) # (B, T, N_EMBED)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, N_EMBED)
        x = tok_emb + pos_emb # (B, T, N_EMBED), positional embeddings are broadcasted across batch
        x = self.blocks(x) # (B, T, N_EMBED)
        x = self.ln_f(x) # (B, T, N_EMBED)
        logits = self.lm_head(x) # (B, T, VOCAB_SIZE)

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
    model = GPTLanguageModel()
    model.to(device)
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
    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    generation = model.generate(inputs, max_new_tokens=200)
    print("\nSample generation after training")
    print("--------------------------------")
    print("".join(decode(generation.squeeze().tolist())))