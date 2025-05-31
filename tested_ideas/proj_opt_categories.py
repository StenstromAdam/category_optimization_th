import numpy as np
from numba import njit, prange
from sklearn.metrics import pairwise_distances

'''
This file is mainly for demonstration of tested ideas. It's an implementation of projected gradient descent for the unit-hypersphere
given input data and initial guess.
'''

def proj_grad_desc(compute_gradient, x0, weight_matrix, lr=0.001, steps=100):
    '''
    Optimizes the points x0 based on the weight matrix.
    :param compute_gradient: The gradient to be used in the calculations.
    :param x0: The initial points of the calculations.
    :param weight_matrix: A matrix defnining relations between points in x0 for determining optimal placement.
    :param lr: The learn rate of the method (default 0.0001).
    :param steps: Number of steps used in calculations (default 1000)
    '''
    x = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    for _ in range(steps):
        x -= lr * compute_gradient(x, weight_matrix)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

@njit(parallel=True, fastmath=True)
def compute_gradient(x, weight_matrix):
    '''
    The gradient of the objective function. Used to calculated the steepest descent direction for next iteration.
    :param x: Current placement of lower-dimensional embeddings.
    :param weight_matrix: A matrix defnining relations between points in x0 for determining optimal placement.
    diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :] 
    dists = np.linalg.norm(diffs, axis=-1, keepdims=True) + 1e-8
    grad = -2 * np.sum(weight_matrix[..., np.newaxis] * diffs / dists, axis=1)
    '''
    n, d = x.shape
    grad = np.zeros((n, d))
    for i in prange(n):
        for j in range(n):
            if weight_matrix[i,j] == 0:
                continue
            diff = x[i] - x[j]
            dist = np.sqrt(np.sum(diff ** 2)) + 1e-8
            grad[i] -= weight_matrix[i, j] * (diff / dist)
    return grad

def projected_gradient(input_data, initial_guess):
    cosine_dissimilarity = pairwise_distances(input_data, metric='cosine')
    weight_matrix = cosine_dissimilarity
    optimal_x = proj_grad_desc(compute_gradient, initial_guess, weight_matrix)
    return optimal_x

