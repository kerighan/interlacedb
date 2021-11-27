import numpy as np
from tqdm import tqdm

# k = 24
# series = 1/(2**np.arange(1, k + 1))
# print(series)
# p = series.sum() + (1/2)**k

# expected = (series * np.arange(1, k+1)).sum() + (k+1)*(1/2)**k
# print(expected)
# # print(series)
N = 1000000
for i in tqdm(range(N)):
    # f'0b{i:04b}'
    bin(i)[-4:]
i+=1
print(bin(i))