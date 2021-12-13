from sqlitedict import SqliteDict
from tqdm import tqdm

from interlacedb import InterlaceDB

with InterlaceDB(
    "test.db",
    flag="n",
    blob_protocol="pickle",
) as db:
    node = db.create_dataset("node", key="U15", value="blob")


N = 10000
block_id = node.new_block(N)
for i in tqdm(range(N)):
    node[block_id, i] = {"key": f"test_{i}", "value": [0] * 200}


db = SqliteDict("test.sqlite", flag="n", autocommit=True)
for i in tqdm(range(N)):
    db[f"test_{i}"] = [0] * 200

for i in tqdm(range(N)):
    db[f"test_{i}"] = [1] * 201

# for i in tqdm(range(N)):
#     db[f"test_{i}"] = [2] * 50
