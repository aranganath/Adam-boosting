import torch
from quasiadam import quasiAdam
import matplotlib.pyplot as plt

x = torch.tensor([-1.0, 1.0], requires_grad=True)

f = lambda x: (x[1] - x[0]**2)**2 + (1-x[0])**2

optimizer = quasiAdam([x], lr=1e-2)

iterates = []

for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
    print(y.item())

    # Save the values of 'x'
    iterates.append(x.detach().numpy())



