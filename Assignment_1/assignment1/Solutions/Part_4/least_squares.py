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
    #print(b)
    #print('x\'s shape is {}'.format(x.shape))
    #print('A\'s shape is {}'.format(A.shape))
    #A = A.T
    #print('A.T\'s shape is {}'.format(A.shape))
    #print((A@x).shape)
    #print(A@x - b)
    #print('x is :\n{}'.format(x)) 
    loss = 0
    for i in range(5000):
       loss+= ( th.norm((A[i,:]@x - b)) )**2 
    return loss
    #return ( th.norm((A@x - b)) ) **2

class LeastSquaresContainer(nn.Module):
    def __init__(self, n):
        super().__init__()
        x = th.tensor(np.random.uniform(low=1, high =13, size=20), dtype=th.float32)
        self.x = nn.Parameter(x)

    def loss(self, A, b):
        return loss_function(A, self.x, b)

def least_squares_approx(A, b, lr=0.01, epochs=2):
    #m, n = A.shape
    #estimator = LeastSquaresContrainer(n=2)
    #estimator = LeastSquaresContainer(n=b.shape[0])
    #print('Getting shape: {}'.format(20))
    estimator = LeastSquaresContainer(n=b)
    loss_curve = []
    optimizer = optim.SGD(estimator.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = estimator.loss(A, b)
        loss.backward()
        loss_curve.append(loss.item())
        optimizer.step()
    
    #plot_loss_curve(loss_curve)
    return estimator

estimator = least_squares_approx(E, d)
print(estimator.x)
values = estimator.x
print(values.shape)
print(values.max())
#print(values[:,0])
#print(values[:,0].sum())
#print(values.max())
#print(values.min())
#print(estimator.loss(E,d))
