import numpy as np

k = 1  # hashing functions
n = 5000  # number of elements
m = 10 * n  # filter size

p = (1 - np.exp(-k * n / m))**k
print(p)
