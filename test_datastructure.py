from random import randint

import numpy as np
from tqdm import tqdm

from interlacedb import InterlaceDB
from interlacedb.datastructure import Dict, FracTable, LayerTable


def test_dict():
    N = 1000000

    db = Dict("dict.db", size=10000)
    for i in tqdm(range(N)):
        db[f"test_{i}"] = {"test": i}

    for i in tqdm(range(N)):
        db[f"test_{i}"]

    for _ in tqdm(db, total=N):
        pass


def test_layertable():
    N = 100000
    with InterlaceDB("test_2.db", flag="n") as db:
        node = db.create_dataset(
            "node", key="U15", value="uint64")
        nodes = LayerTable(
            node, key="key",
            p_init=17, growth_factor=2, probe_factor=.1,
            cache_len=100000,
            n_bloom_filters=10)  # 188.7, 165.6
        db.create_datastructure("nodes", nodes)

    for i in tqdm(range(N)):
        nodes[f"test_{i}"] = {"value": i}

    for i in tqdm(range(N)):
        nodes[f"test_{i}"]

    for i in tqdm(range(N)):
        nodes[f"test_{i}"]

    print(nodes["test_55"])


def test_fractable():
    with InterlaceDB("test.db", flag="n") as db:
        node = db.create_dataset("node", key="U15", value="uint64")
        nodes = db.create_datastructure("nodes",
                                        FracTable(node, "key", p_init=9))

    N = 500000
    for i in tqdm(range(N)):
        nodes.insert({"key": f"test_{i}", "value": i})
    for i in tqdm(range(N)):
        nodes.lookup(f"test_{i}")


# test_dict()
test_layertable()
# test_fractable()
