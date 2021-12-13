import random

import numpy as np
from numpy.core.shape_base import block
from tqdm import tqdm

from interlacedb import InterlaceDB

with InterlaceDB("test.db", flag="n") as db:
    db.create_header(n_nodes="uint64", n_edges="uint64", block_id="uint64")
    dset = db.create_dataset("nodes", key="U20", value="uint64")

db.header["block_id"] = 5

N = 100000
vector = np.arange(200).astype(np.float32)
block_id = dset.new_block(N)

# for i in tqdm(range(N), desc="fill with data")

# db = InterlaceDB("test.db")
# block_id = db.header["block_id"]
# print(block_id)
