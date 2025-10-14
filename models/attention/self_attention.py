import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


B, T, C = 4, 8, 32

x = torch.randn((B, T, C))

# single headed self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)  # (B,T,16)
q = query(x)  # (B,T,16)

k_transpose = rearrange(k, "b t h -> b h t")  # (B,16,T)
wei = q @ k_transpose  # (B,T,16) @ (B,16,T) => (B,T,T)

tril = torch.tril(torch.ones((T, T)))
# wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
# out = wei @ x

print(out.shape)
