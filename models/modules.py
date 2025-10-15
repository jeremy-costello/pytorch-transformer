import torch
from torch import nn
from models.attention.self_attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    def __init__(
            self,
            n_embd: int,
            dropout: float
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(
            self,
            x: torch.Tensor
    ):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            n_head: int,
            n_embd: int,
            context_length: int,
            dropout: float
    ):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadSelfAttention(
            n_head=n_head,
            head_size=head_size,
            n_embd=n_embd,
            context_length=context_length,
            dropout=dropout
        )
        self.ffwd = FeedForward(
            n_embd=n_embd,
            dropout=dropout
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(
            self,
            x: torch.Tensor
    ):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
