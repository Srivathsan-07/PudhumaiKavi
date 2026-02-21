import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # no. of independent examples to train on in one run parallely
block_size = 256 # size of context window
max_iters = 5000 # no of training iterations
eval_interval = 500 # no of iteration after which to evaluate the model
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

