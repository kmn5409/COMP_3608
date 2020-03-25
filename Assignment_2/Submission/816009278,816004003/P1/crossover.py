import numpy as np 


def single_point(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    print(point)
    child = np.zeros_like(parent1)
    child[:point] = parent1[:point]
    child[point:] = parent2[point:]
    return child


def interpolation(parent1, parent2):
    lambda_point = np.random.uniform()
    lambda_point = 0.4
    return   (1 - lambda_point) * parent1 + lambda_point * parent2

def uniform(parent1, parent2):
    selection = np.random.randint(0, 2, size=len(parent1))
    selection = np.asarray([0.5,0.6,0.1,0.2,0.99])
    neg_selection = 1 - selection
    child = parent1 * selection + parent2 * neg_selection
    return child 
