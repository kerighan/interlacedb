from random import randint

import numpy as np
from tqdm import tqdm

from interlacedb import InterlaceDB
from interlacedb.datastructure import LayeredHashTable

N = 100000

with InterlaceDB("test.db", flag="n") as db:
    node = db.create_dataset(
        "node", key="U15", value="blob")
    nodes = LayeredHashTable(
        node, key="key", p_init=7, branching_factor=3, probe_factor=.5,
        n_bloom_filters=10)  # 188.7, 165.6
    db.create_datastructure("nodes", nodes)

# db = InterlaceDB("test.db")
# nodes = db.datastructures["node_htable"]

for i in tqdm(range(N)):
    nodes[f"test_{i}"] = {"value": {"test": i}}

for i in tqdm(range(N)):
    nodes[f"test_{i}"]
print(nodes["test_55"])

# for i in tqdm(range(N)):
#     f"test_{i}" in nodes
