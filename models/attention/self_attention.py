import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        head_size: int,
        n_embd: int,
        context_length: int,
        dropout: float
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones((context_length, context_length))
            )
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor
    ):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        k_transpose = rearrange(k, "b t h -> b h t")
        
        # compute attention scores ("affinities")
        wei = q @ k_transpose * C**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            n_head: int,
            head_size: int,
            n_embd: int,
            context_length: int,
            dropout: float
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    head_size=head_size,
                    n_embd=n_embd,
                    context_length=context_length,
                    dropout=dropout
                )
                for _ in range(n_head)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
            self,
            x: torch.Tensor
    ):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        out = self.dropout(self.proj(out))
        return out
