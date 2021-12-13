from sqlitedict import SqliteDict
from tqdm import tqdm

from interlacedb import InterlaceDB
from interlacedb.datastructure import LayeredHashTable

N = 10000000

# with InterlaceDB("test.db", flag="n") as db:
#     node = db.create_dataset("node", key="U20", value="uint64")
#     node_htable = LayeredHashTable(
#         node, key="key",
#         p_init=5, branching_factor=4, probe_factor=1)
#     db.create_datastructure("node_htable", node_htable)

db = InterlaceDB("test.db")
node_htable = db.datastructures["node_htable"]

# insert data
for i in tqdm(range(N)):
    node_htable.insert({"key": f"test_{i}", "value": i})

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
