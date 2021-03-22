import numpy as np

def sphere(vector):
    z = np.sum(vector**2)
    return z

def ackley(vector):
    x = vector[0]
    y = vector[1]
    z = -20 * np.exp((-0.2*np.sqrt(0.5*(x**2 + y**2)))) - np.exp(0.5 * (np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.exp(1) + 20
    return z