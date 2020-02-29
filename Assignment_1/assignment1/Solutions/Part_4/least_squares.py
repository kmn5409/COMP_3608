import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


E = np.load('../../E.npy')
d = np.load('../../d.npy')
E = th.tensor(E, dtype=th.float32)
d = th.tensor(d, dtype=th.float32)
print('E is :\n{}\n d is :\n{}'.format(E,d))
print('E\'s shape is {}'.format(E.shape))
print('d\'s shape is {}'.format(d.shape))
print('d\'s biggest value is {}'.format(d.max()))
print('d\'s smallest value is {}'.format(d.min()))
print(E[0,:])


def plot_loss_curve(loss_curve):
    plt.plot(list(range(len(loss_curve))), loss_curve)
    plt.show()

def loss_function(A, x, b):
    return ( th.norm((A@x - b)) ) **2

class LeastSquaresContainer(nn.Module):
    def __init__(self, n):
        super().__init__()
        x = th.tensor(np.random.random(n), dtype=th.float32)
        self.x = nn.Parameter(x)

    def loss(self, A, b):
        return loss_function(A, self.x, b)

def least_squares_approx(A, b, lr=0.00001, epochs=1000):
    estimator = LeastSquaresContainer(n=A.shape[1])
    loss_curve = []
    optimizer = optim.SGD(estimator.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = estimator.loss(A, b)
        loss.backward()
        loss_curve.append(loss.item())
        optimizer.step()
    
    plot_loss_curve(loss_curve)
    return estimator

estimator = least_squares_approx(E, d)
values = estimator.x
print(values)
