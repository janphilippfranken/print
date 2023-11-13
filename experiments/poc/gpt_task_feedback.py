from print.utility.base import BaseUtility

"Utility Function As Code"
import numpy as np
import io
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

def utility_func(algorithm_str: str):
    """
    Trains a GPT-like language model called NanoGPT. Returns the number of correct predictions,
    along with the output of print statements and error messages.
    If an error occurs, the function breaks the loop and returns immediately.
    """
    n_tests = 3
    average_correct = 0
    all_outputs = ""
    string_io = io.StringIO() 
    
    try:
        exec(algorithm_str, globals())
    except Exception as e:
        return 0, f"Execution Error: {e}"
    
    for _ in range(n_tests):
        # hyperparameters
        batch_size = 8 # how many independent sequences will we process in parallel?
        block_size = 16 # what is the maximum context length for predictions?
        max_iters = 50
        eval_interval = 10
        learning_rate = 1e-3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        eval_iters = 200
        n_embd = 32
        n_head = 4
        n_layer = 3
        dropout = 0.2
        # ------------
        torch.manual_seed(1337)
        
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] 
        decode = lambda l: ''.join([itos[i] for i in l]) 

        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data)) 
        train_data = data[:n]
        val_data = data[n:]

        # data loading
        def get_batch(split):
            # generate a small batch of data of inputs x and targets y
            data = train_data if split == 'train' else val_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i:i+block_size] for i in ix])
            y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out



        class MultiHeadAttention(nn.Module):
            """ multiple heads of self-attention in parallel """

            def __init__(self, num_heads, head_size):
                super().__init__()
                self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
                self.proj = nn.Linear(n_embd, n_embd)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                out = torch.cat([h(x) for h in self.heads], dim=-1)
                out = self.dropout(self.proj(out))
                return out

        class FeedFoward(nn.Module):
            """ a simple linear layer followed by a non-linearity """

            def __init__(self, n_embd):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.ReLU(),
                    nn.Linear(4 * n_embd, n_embd),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                return self.net(x)

        class Block(nn.Module):
            """ Transformer block: communication followed by computation """

            def __init__(self, n_embd, n_head):
                # n_embd: embedding dimension, n_head: the number of heads we'd like
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

        # super simple bigram model
        class NanoGPT(nn.Module):

            def __init__(self):
                super().__init__()
                # each token directly reads off the logits for the next token from a lookup table
                self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
                self.position_embedding_table = nn.Embedding(block_size, n_embd)
                self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
                self.ln_f = nn.LayerNorm(n_embd) # final layer norm
                self.lm_head = nn.Linear(n_embd, vocab_size)

            def forward(self, idx, targets=None):
                B, T = idx.shape

                # idx and targets are both (B,T) tensor of integers
                tok_emb = self.token_embedding_table(idx) # (B,T,C)
                pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
                x = tok_emb + pos_emb # (B,T,C)
                x = self.blocks(x) # (B,T,C)
                x = self.ln_f(x) # (B,T,C)
                logits = self.lm_head(x) # (B,T,vocab_size)

                if targets is None:
                    loss = None
                else:
                    B, T, C = logits.shape
                    logits = logits.view(B*T, C)
                    targets = targets.view(B*T)
                    loss = F.cross_entropy(logits, targets)

                return logits, loss

            def generate(self, idx, max_new_tokens):
                # idx is (B, T) array of indices in the current context
                for _ in range(max_new_tokens):
                    # crop idx to the last block_size tokens
                    idx_cond = idx[:, -block_size:]
                    # get the predictions
                    logits, loss = self(idx_cond)
                    # focus only on the last time step
                    logits = logits[:, -1, :] # becomes (B, C)
                    # apply softmax to get probabilities
                    probs = F.softmax(logits, dim=-1) # (B, C)
                    # sample from the distribution
                    idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                    # append sampled index to the running sequence
                    idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                return idx

        model = NanoGPT()
        m = model.to(device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters()), "params")

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


        # Redirect standard output and standard error
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = string_io

        try:
            for iter in range(max_iters):

                xb, yb = get_batch('train')

                # evaluate the loss
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                correct = -torch.exp(loss.item()).numpy()
        except Exception as e:
            break  # Break the loop on error
        finally:
            all_outputs += string_io.getvalue()
            string_io.seek(0)  # Clear buffer for next iteration
            string_io.truncate()
            sys.stdout, sys.stderr = old_stdout, old_stderr


        average_correct += correct / n_tests
    
    return average_correct, all_outputs

"Utility Function As String"
utility_str = """"
import numpy as np
import io
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

def utility_func(algorithm_str: str):
    ""
    Trains a GPT-like language model called NanoGPT. Returns the number of correct predictions,
    along with the output of print statements and error messages.
    If an error occurs, the function breaks the loop and returns immediately.
    ""
    n_tests = 3
    average_correct = 0
    all_outputs = ""
    string_io = io.StringIO() 
    
    try:
        exec(algorithm_str, globals())
    except Exception as e:
        return 0, f"Execution Error: {e}"
    
    for _ in range(n_tests):
        # hyperparameters
        batch_size = 8 # how many independent sequences will we process in parallel?
        block_size = 16 # what is the maximum context length for predictions?
        max_iters = 50
        eval_interval = 10
        learning_rate = 1e-3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        eval_iters = 200
        n_embd = 32
        n_head = 4
        n_layer = 3
        dropout = 0.2
        # ------------
        torch.manual_seed(1337)
        
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] 
        decode = lambda l: ''.join([itos[i] for i in l]) 

        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9*len(data)) 
        train_data = data[:n]
        val_data = data[n:]

        # data loading
        def get_batch(split):
            # generate a small batch of data of inputs x and targets y
            data = train_data if split == 'train' else val_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i:i+block_size] for i in ix])
            y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out



        class MultiHeadAttention(nn.Module):
            " multiple heads of self-attention in parallel "

            def __init__(self, num_heads, head_size):
                super().__init__()
                self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
                self.proj = nn.Linear(n_embd, n_embd)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                out = torch.cat([h(x) for h in self.heads], dim=-1)
                out = self.dropout(self.proj(out))
                return out

        class FeedFoward(nn.Module):
            " a simple linear layer followed by a non-linearity "

            def __init__(self, n_embd):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.ReLU(),
                    nn.Linear(4 * n_embd, n_embd),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                return self.net(x)

        class Block(nn.Module):
            " Transformer block: communication followed by computation "

            def __init__(self, n_embd, n_head):
                # n_embd: embedding dimension, n_head: the number of heads we'd like
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

        # super simple bigram model
        class NanoGPT(nn.Module):

            def __init__(self):
                super().__init__()
                # each token directly reads off the logits for the next token from a lookup table
                self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
                self.position_embedding_table = nn.Embedding(block_size, n_embd)
                self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
                self.ln_f = nn.LayerNorm(n_embd) # final layer norm
                self.lm_head = nn.Linear(n_embd, vocab_size)

            def forward(self, idx, targets=None):
                B, T = idx.shape

                # idx and targets are both (B,T) tensor of integers
                tok_emb = self.token_embedding_table(idx) # (B,T,C)
                pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
                x = tok_emb + pos_emb # (B,T,C)
                x = self.blocks(x) # (B,T,C)
                x = self.ln_f(x) # (B,T,C)
                logits = self.lm_head(x) # (B,T,vocab_size)

                if targets is None:
                    loss = None
                else:
                    B, T, C = logits.shape
                    logits = logits.view(B*T, C)
                    targets = targets.view(B*T)
                    loss = F.cross_entropy(logits, targets)

                return logits, loss

            def generate(self, idx, max_new_tokens):
                # idx is (B, T) array of indices in the current context
                for _ in range(max_new_tokens):
                    # crop idx to the last block_size tokens
                    idx_cond = idx[:, -block_size:]
                    # get the predictions
                    logits, loss = self(idx_cond)
                    # focus only on the last time step
                    logits = logits[:, -1, :] # becomes (B, C)
                    # apply softmax to get probabilities
                    probs = F.softmax(logits, dim=-1) # (B, C)
                    # sample from the distribution
                    idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                    # append sampled index to the running sequence
                    idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                return idx

        model = NanoGPT()
        m = model.to(device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters()), "params")

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


        # Redirect standard output and standard error
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = string_io

        try:
            for iter in range(max_iters):

                xb, yb = get_batch('train')

                # evaluate the loss
                logits, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                correct = -torch.exp(loss.item()).numpy()
        except Exception as e:
            break  # Break the loop on error
        finally:
            all_outputs += string_io.getvalue()
            string_io.seek(0)  # Clear buffer for next iteration
            string_io.truncate()
            sys.stdout, sys.stderr = old_stdout, old_stderr


        average_correct += correct / n_tests
    
    return average_correct, all_outputs"""

algorithm_str="""

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        n_embd = 32
        head_size = 4
        block_size = 16
        dropout = 0.2
        self.key = nn.Linear(head_size, n_embd, bias=False)
        self.query = nn.Linear(head_size, n_embd, bias=False)
        self.value = nn.Linear(head_size, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(n_embd, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,C,T = x.shape
        k = self.key(x)  
        q = self.query(x) 
        wei = q @ k.transpose(-1,-2) * T**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v.transpose(-1,-2)
        return out"""

class GPT_Task():

    def __init__(
        self, 
        utility, 
    ) -> None:
        self.utility = utility
        self.solutions = []
    def add_solution(self, solution):
        self.solutions.append(solution)
    def get_solution(self, i: int = -1):
        return self.solutions[i]

"Utility Object"
utility = BaseUtility(utility_id="gpt_feedback")
utility.add_utility(utility_func, utility_str)


gpt_task_feedback = GPT_Task(utility)
gpt_task_feedback.add_solution(algorithm_str)