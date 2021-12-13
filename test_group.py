from numpy.core.shape_base import block
from sqlitedict import SqliteDict
from tqdm import tqdm

from interlacedb import InterlaceDB

with InterlaceDB("test.db", flag="n") as db:
    node = db.create_dataset("node", key="U20", value="uint64")
    node_group = db.create_group("node_group", node, group_value="uint64")


block_id = node_group.new_block(100)

node_group[block_id, "group_value"] = 5

for i in range(100):
    node_group[block_id, i] = {"key": f"key_{i}", "value": i}

for i in range(100):
    print(node_group[block_id, i])
