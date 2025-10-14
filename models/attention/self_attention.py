import torch
from torch.nn import functional as F


B, T, C = 4, 8, 32

x = torch.randn((B, T, C))

tril = torch.tril(torch.ones((T, T)))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
out = wei @ x

print(out.shape)
