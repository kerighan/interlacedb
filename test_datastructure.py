from interlacedb import InterlaceDB
from interlacedb.datastructure import HashTable, HashTree
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import os

if os.path.exists("test.db"):
    os.remove("test.db")

with InterlaceDB("test.db") as db:
    node = db.create_dataset("node", value="int")
    edge = db.create_dataset("edge", source="int", target="int")

    htable_node = db.create_datastructure(HashTree(node))
    htable_edge = db.create_datastructure(HashTree(edge))


N = 2**16
edges = []
for i in tqdm(range(N)):
    key = f"key_{i}"
    htable_node.insert(key, {"value": i})

    u = random.randint(0, N-1)
    v = random.randint(0, N-1)
    key = f"{u}_{v}"
    htable_edge.insert(key, {"source": u, "target": v})
    edges.append((u, v))


keys = [f"key_{i}" for i in range(N)]
for key in tqdm(keys, desc="get node"):
    htable_node.lookup(key)

for u, v in tqdm(edges, desc="get edge"):
    key = f"{u}_{v}"
    htable_edge.lookup(key)
