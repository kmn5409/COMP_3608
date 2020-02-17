import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


E = np.load('../../E.npy')
d = np.load('../../d.npy')
E = th.tensor(E, dtype=th.float32)
d = th.tensor(d, dtype=th.float32)
print('{}\n\n{}'.format(E,d))
print('E\'s shape is {}'.format(E.shape))
print('d\'s shape is {}'.format(d.shape))
print(E[0,:])


def plot_loss_curve(loss_curve):
    plt.plot(list(range(len(loss_curve))), loss_curve)
    plt.show()

def loss_function(A, x, b):
    #print(b)
    #print('x\'s shape is {}'.format(x.shape))
    #print('A\'s shape is {}'.format(A.shape))
    #A = A.T
    #print('A.T\'s shape is {}'.format(A.shape))
    #print((A@x).shape)
    #print(A@x - b)
    
    return (th.norm((A@x - b))) **2

class LeastSquaresContainer(nn.Module):
    def __init__(self, n):
        super().__init__()
        x = th.tensor(np.random.random(n), dtype=th.float32)
        self.x = nn.Parameter(x)

    def loss(self, A, b):
        return loss_function(A, self.x, b)

def least_squares_approx(A, b, lr=0.01, epochs=200):
    #m, n = A.shape
    #estimator = LeastSquaresContrainer(n=2)
    estimator = LeastSquaresContainer(n=b.shape[0])
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

estimator = least_squares_approx(E[:,0], d)
