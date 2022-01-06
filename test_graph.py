import os
import random

from tqdm import tqdm

from interlacedb.beta import Graph

try:
    os.remove("test.db")
except FileNotFoundError:
    pass

G = Graph("test.db")

for i in tqdm(range(1, 1000000)):
    G.add_edge(0, i)

for _ in tqdm(range(1)):
    G.neighbors(0)

# N = 5000
# M = 50*N

# for i in tqdm(range(N)):
#     G.add_node(i)

# for i in tqdm(range(M)):
#     u = random.randint(0, N-1)
#     v = random.randint(0, N-1)
#     G.add_edge(u, v)

# for i in tqdm(range(N)):
#     G.neighbors(i)
#     G.predecessors(i)

# print(G.header["n_nodes"])
