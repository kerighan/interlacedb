import numpy as np
from numpy.core.shape_base import block
from tqdm import tqdm

from interlacedb import InterlaceDB

# with InterlaceDB("test.db", flag="n") as db:
#     db.create_header(n_nodes="uint64", n_edges="uint64", block_id="uint64")
#     dset = db.create_dataset("nodes", key="U20", value="uint64",
#                              vector="(200,)float32")
#     # dstruct = db.create_datastructure()


# N = 100000
# vector = np.arange(200).astype(np.float32)
# block_id = dset.new_block(N)
# for i in tqdm(range(N), desc="fill with data"):
#     dset[block_id, i] = {"key": f"salut_{i}", "value": i, "vector": vector}


# for i in tqdm(range(N), desc="update value"):
#     dset[block_id, i, "key"] = "toto"

# for i in tqdm(range(N), desc="get row"):
#     dset[block_id, i]

# for i in tqdm(range(N), desc="get specific field"):
#     dset[block_id, i, "key"]

# db.header["block_id"] = block_id

db = InterlaceDB("test.db")
block_id = db.header["block_id"]
dset = db.datasets["nodes"]
print(dset[660, 55])
