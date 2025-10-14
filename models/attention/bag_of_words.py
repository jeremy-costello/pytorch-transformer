import torch
from torch.nn import functional as F


B, T, C = 4, 8, 2

x = torch.randn((B, T, C))

# VERSION 1
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, dim=0)

print(xbow.mean().cpu().item())

# VERSION 2
tril = torch.tril(torch.ones((T, T)))
wei = tril / torch.sum(tril, dim=-1, keepdim=True)
xbow2 = wei @ x

print(xbow2.mean().cpu().item())

# VERSION 3
tril = torch.tril(torch.ones((T, T)))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x

print(xbow3.mean().cpu().item())

print(torch.allclose(xbow, xbow2))
print(torch.allclose(xbow, xbow3))
print(torch.allclose(xbow2, xbow3))
