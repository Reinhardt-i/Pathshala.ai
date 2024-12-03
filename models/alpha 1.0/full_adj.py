import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer

# Set the random seed for reproducibility
seed = 1337
torch.manual_seed(seed)

tokenizer = bangla_custom_tokenizer()
tokenizer.pad_token = tokenizer.eos_token  # Ensure tokenizer has a padding token

# Read and encode the dataset using the tokenizer
with open('all.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset in characters:", len(text))

# Encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print("Encoded dataset shape:", data.shape)

# Hyperparameters
batch_size = 512  # Batch size
block_size = 1024  # Maximum context length for predictions
max_iters = 100000  # Total training iterations
eval_interval = 1000  # Evaluation interval
learning_rate = 2.5e-4  # Learning rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
eval_iters = 200  # Evaluation iterations
n_embd = 768  # Embedding dimension
n_head = 12  # Number of attention heads
n_layer = 12  # Number of transformer blocks
dropout = 0.1  # Dropout rate
vocab_size = tokenizer.vocab_size  # Vocabulary size (50,257 tokens)

# Calculate warm-up and total steps for the learning rate scheduler
num_warmup_steps = int(0.001 * max_iters)  # Warm-up over the first 0.1% of total steps

# Train and validation splits
n = int(0.9 * len(data))  # First 90% for training, rest for validation
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # Scaled dot-product attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate heads
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Activation function: GELU
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection
        x = x + self.ffwd(self.ln2(x))  # Residual connection
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for loss computation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize the model
model = BigramLanguageModel().to(device)

# Print the number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Optimizer with specified betas and epsilon
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

# Learning rate scheduler
def lr_lambda(current_step):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, float(max_iters - current_step) / float(max(1, max_iters - num_warmup_steps))
    )

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop
for iter in range(max_iters):

    # Evaluate loss at intervals
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

    # Get a batch of data
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass and optimization step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update the learning rate

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device).fill_(tokenizer.bos_token_id)
generated_tokens = model.generate(context, max_new_tokens=2000, temperature=0.8, top_k=200)[0].tolist()
print(tokenizer.decode(generated_tokens))


