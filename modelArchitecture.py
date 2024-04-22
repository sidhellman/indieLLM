import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the environment variable to use only GPU 1

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import sys
import time
import math
import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer

# Hyperparameters
class Config:
    batch_size = 16
    block_size = 128  # Increased block size
    max_iters = 20000
    eval_interval = 1000
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 256
    n_head = 16
    n_layer = 12
    dropout = 0.1
    data_path = '/root/train_txt/cleaned_input.txt'
    model_path = 'model.pt'

config = Config()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=config.data_path)
parser.add_argument('--model_path', type=str, default=config.model_path)
parser.add_argument('--batch_size', type=int, default=config.batch_size)
parser.add_argument('--block_size', type=int, default=config.block_size)
parser.add_argument('--max_iters', type=int, default=config.max_iters)
parser.add_argument('--eval_interval', type=int, default=config.eval_interval)
parser.add_argument('--learning_rate', type=float, default=config.learning_rate)
parser.add_argument('--eval_iters', type=int, default=config.eval_iters)
parser.add_argument('--n_embd', type=int, default=config.n_embd)
parser.add_argument('--n_head', type=int, default=config.n_head)
parser.add_argument('--n_layer', type=int, default=config.n_layer)
parser.add_argument('--dropout', type=float, default=config.dropout)

# Extract only the relevant arguments
args = parser.parse_known_args(sys.argv[1:])[0]
config.__dict__.update(args.__dict__)

torch.manual_seed(1409)

# Load and preprocess data
with open(config.data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Create a directory for saving the tokenizer files
os.makedirs("tokenizer_files", exist_ok=True)

# Create a Byte-Level BPE tokenizer
tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

# Train the tokenizer on the text data
tokenizer.train_from_iterator(
    [text.split('\n')],
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

# Save the trained tokenizer files
tokenizer.save_model("tokenizer_files", "byte_level_bpe")

# Encode the text using the BPE tokenizer
encoded_text = tokenizer.encode(text).ids

# Split the encoded text into train and validation sets
train_data = encoded_text[:int(0.9*len(encoded_text))]
val_data = encoded_text[int(0.9*len(encoded_text)):]

# Update the vocabulary size
vocab_size = tokenizer.get_vocab_size()

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.tensor(data[i:i+config.block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+config.block_size+1]) for i in ix])
    return x.to(config.device), y.to(config.device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Increased feed-forward dimension
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=config.device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
model = model.to(config.device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def train():
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), config.model_path)
        
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    
def load_model():
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    print("Model loaded from", config.model_path)
    
def generate_text(prompt):
    # Load the trained tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "tokenizer_files/byte_level_bpe-vocab.json",
        "tokenizer_files/byte_level_bpe-merges.txt",
        add_prefix_space=True,
    )
    
    encoded_prompt = tokenizer.encode(prompt).ids
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=config.device).unsqueeze(0)
    generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

if __name__ == "__main__":
    train()
    load_model()
    prompt = "Ram went to\n"
    print(generate_text(prompt))
