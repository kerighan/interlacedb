import time

from tqdm import tqdm

from interlacedb import InterlaceDB

with InterlaceDB("test.db", flag="n") as db:
    edge = db.create_dataset("edge", value="uint64")

N = 100000
block_id = edge.new_block(N)
for i in tqdm(range(N)):
    if i == 2:
        continue
    edge[block_id, i] = dict(value=i)

start = time.time()
d = edge[block_id, :N, "value"]
print(time.time() - start)
