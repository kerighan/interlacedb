from random import randint

import numpy as np
from tqdm import tqdm

from interlacedb import InterlaceDB
from interlacedb.datastructure import Dict, LayeredTable


def test_dict():
    N = 100000

    db = Dict("dict.db", size=100000)
    for i in tqdm(range(N)):
        db[f"test_{i}"] = {"salutos": i}

    for i in tqdm(range(N)):
        db[f"test_{i}"]

    for key, value in tqdm(db, total=N):
        pass


def test_layeredtable():
    N = 100000
    with InterlaceDB("test.db", flag="n") as db:
        node = db.create_dataset(
            "node", key="U15", value="blob")
        nodes = LayeredTable(
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

    for i in tqdm(range(N)):
        f"test_{i}" in nodes


test_dict()
