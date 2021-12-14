import numpy as np

k = 1  # hashing functions
n = 500000  # number of elements
m = 200 * n  # filter size

p = (1 - np.exp(-k * n / m))**k
print(p)
