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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # interesting how the output dimension is vocab_size
        self.embedding_table = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
        # this is a very simple model
        # the embedding can be directly interpreted as the logits (predictions for next token)

    def forward(self, x, y=None):
        # x: (B, T)
        # y: (B, T)
        # embedding table essentially replaces each index with its corresponding embedding
        logits = self.embedding_table(x) # logits: (B, T, C)
        # calculate loss:
        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, x, max_new_tokens):
        """
        quick note about this method:
        this is a simple bigram model, so it only needs the immediate previous
        character to predict the next token
        however this method's implementation feeds the entire previous context,
        and then we just extract the last prediction
        this is obviously inefficient, as we could simply pass the most recent token,
        to predict the next one
        however this method's implementation will scale to more complex architectures,
        which actually care about context length :)        
        """
        # x: (B, T)
        for _ in range(max_new_tokens):
            # get predictions
            logits, _ = self(x) # logits: (B, C, T)
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