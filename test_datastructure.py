import numpy as np
from sqlitedict import SqliteDict
from tqdm import tqdm

from interlacedb import InterlaceDB
from interlacedb.datastructure import LayeredHashTable

N = 100000

# with InterlaceDB("test.db", flag="n") as db:
#     node = db.create_dataset(
#         "node", key="U15", value="uint64")
#     node_htable = LayeredHashTable(
#         node, key="key", p_init=10, branching_factor=2, probe_factor=.25)
#     db.create_datastructure("node_htable", node_htable)


# # insert data
# for i in tqdm(range(N)):
#     node_htable.insert({"key": f"test_{i}", "value": i})

# for i in tqdm(range(N)):
#     node_htable.lookup(f"test_{i}")

db = SqliteDict("test.sqlite", autocommit=True)
for i in tqdm(range(N)):
    db[f"test_1"] = np.arange(i, 200+i)
# for i in tqdm(range(N)):
#     db[f"test_{i}"]
