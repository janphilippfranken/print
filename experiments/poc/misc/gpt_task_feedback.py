from print.utility.base import BaseUtility

"Utility Function As Code"
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.poc.misc.train_gpt import train_gpt

def utility_func(algorithm_str: str):
    """
    Evaluates the performance of the 'Head' class defined in algorithm_str.
    The class will be used in a GPT language model using the train_gpt function and the loss of the model will depend on the Head.
    If an error occurs, the function breaks the loop and returns negative infinity immediately.
    """
    n_tests = 3
    average_loss = 0
    try:
        exec(algorithm_str, globals())
    except:
        return -torch.inf, ""

    for _ in range(n_tests):
    
        try:
            loss, print_outputs = train_gpt(Head)
        except:
            loss, print_outputs = -torch.inf, ""
            break

        average_loss += loss / n_tests
    
    return average_loss, print_outputs

"Utility Function As String"
utility_str = """def utility_func(algorithm_str: str):
    \"\"\"
    Evaluates the performance of the 'Head' class defined in algorithm_str.
    The class will be used in a GPT language model using the train_gpt function and the loss of the model will depend on the Head.
    If an error occurs, the function breaks the loop and returns negative infinity immediately.
    \"\"\"
    n_tests = 3
    average_loss = 0
    try:
        exec(algorithm_str, globals())
    except:
        return -torch.inf, ""

    for _ in range(n_tests):
    
        try:
            loss, print_outputs = train_gpt(Head)
        except:
            loss, print_outputs = -torch.inf, ""
            break

        average_loss += loss / n_tests
    
    return average_loss, print_outputs"""

correct_str="""# hyperparameters
block_size = 8 
n_embd = 32
dropout = 0.2

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out"""

algorithm_str="""# hyperparameters
block_size = 8 
n_embd = 32
dropout = 0.2

class Head(nn.Module):
  
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=True)
        self.query = nn.Linear(n_embd, head_size, bias=True)
        self.value = nn.Linear(head_size, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size + 1)))

        self.dropout = nn.Dropout(dropout + 0.1)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-1,-2) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T+1] == 1, float('-inf'))
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v

        return out"""

class Task():

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

gpt_task_feedback = Task(utility)
gpt_task_feedback.add_solution(algorithm_str)