import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # no. of independent examples to train on in one run parallely
block_size = 256 # size of context window
max_iters = 5000 # no of training iterations
eval_interval = 250 # no of iteration after which to evaluate the model
learning_rate = 3e-3 # gradient descent learing rate
device = 'cuda' if torch.cuda.is_available() else 'cpu' # using cuda if available
eval_iters = 200 # no of iterations of evaluation to be averaged out after
n_embd = 384 # no of charcs the model uses to define a single character 
n_layer = 6 # no of layers in the sequential -- 
n_head = 6 # no of attention heads (must divide n_embd without reminder)
dropout = 0.2 # drop out implemented to 20% of nueral nodes so we don't overfit

torch.manual_seed(1337)

# loading dataset
with open('train_base1.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# creating set of all unique characters in the set
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from chars to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# train test spilts
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch
    # wether train_data or val_data based on split passed
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # create ix a (len(data)-block_size, (,batch_size)) randint matrix to get the ideices
    x = torch.stack([data[i:i+block_size] for i in ix]) # for every i in ix, get data for i till i+bloack_size - x
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # for every i in ix, get data for i+1 till i+block_size+1 - y
    x, y = x.to(device), y.to(device) # store x and y to device
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # chaning model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # averaging the losses across all the eval iterations
    model.train() # reverting model to training mode
    return out

class Head(nn.Module):
    """Single Head of self-attention"""

    def __init__(self, head_size):
        super().__init__() # initializing the super(parent) class
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout =  nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # key layer is being applied to the input x
        q = self.query(x) # query layer is being applied to the input x
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # dividing by the C to scale the dotproduct
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads of logits for the next token from a look-up table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both B, T tensors of integers
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # evaluate the loss every once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss { losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(m.state_dict(), 'wikiGPT.pt')

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = m.generate(context, max_new_tokens=500)[0].tolist()
print(f"Generated {len(generated_indices)} tokens.")
print(decode(generated_indices))

output_text = decode(model.generate(context, max_new_tokens=5000)[0].tolist())


with open('generated_output_wiki.txt', 'w', encoding='utf-8') as f:
    f.write(output_text)