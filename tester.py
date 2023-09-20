import torch
from quasiadam import quasiAdam
from torch.optim import Adam


x = torch.tensor([-20., 1.,-1.,1.], requires_grad=True)
f = lambda x: (torch.sum(x[1::2]-x[::2]**2))**2 + (torch.sum(torch.ones_like(x[::2])-x[::2]))**2


optimizer = quasiAdam([x], lr = 1)
for i in range(500):
    y= f(x)
    optimizer.zero_grad()
    y.backward()
    print(y.item())
    optimizer.step()