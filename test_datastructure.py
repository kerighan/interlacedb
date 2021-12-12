from interlacedb import InterlaceDB
from interlacedb.datastructure import LayeredHashTable
from sqlitedict import SqliteDict
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import os

N = 100000
with InterlaceDB("test.db", flag="n") as db:
    node = db.create_dataset("node", key="U20", value="uint64")
    node_htable = LayeredHashTable(
        node, key="key",
        p_init=10, branching_factor=2, probe_factor=.25)
    db.create_datastructure("node_htable", node_htable)

for i in tqdm(range(N)):
    node_htable.insert({"key": f"test_{i}", "value": i})

# print(node_htable.lookup("test_0"))
# print(node_htable.lookup(f"test_9999"))
# del db

# db = InterlaceDB("test.db")
# node_htable = db.datastructures["node_htable"]
# print(node_htable.tables_id)
for i in tqdm(range(N)):
    try:
        node_htable.lookup(f"test_{i}")
    except ValueError:
        print(i)
        raise ValueError


# db = SqliteDict("test.sqlite", autocommit=True)
# for i in tqdm(range(N)):
#     db[f"test_{i}"] = i
# for i in tqdm(range(N)):
#     db[f"test_{i}"]


# node_htable.insert({"key": "test", "value": 55})
# print(node_htable.lookup("test"))


# print(node_htable.p_last)

# print(node_htable.tables_id)
# print(db.header["node_LHT_tables_id"])
