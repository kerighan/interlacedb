from interlacedb import InterlaceDB
from tqdm import tqdm
from interlacedb.datastructure import FloatingLayerTable
import time

N = 1000000

# with InterlaceDB("test.db", flag="n") as db:
#     node = db.create_dataset("node", key="U15")
#     edge = db.create_dataset("edge", node="U15")
#     edges = FloatingLayerTable(
#         edge, key="node",
#         growth_factor=2, p_init=0,
#         probe_factor=.25, n_bloom_filters=25)
#     db.create_datastructure("edges", edges)

db = InterlaceDB("test.db")
edges = db.datastructures["edges"]

t_id = edges.new_table()
for i in tqdm(range(N)):
    new_t_id = edges.insert(t_id, {"node": f"test_{i}"})
    if new_t_id > t_id:
        t_id = new_t_id

for i in tqdm(range(N)):
    new_t_id = edges.insert(t_id, {"node": f"test_{i}"})
    if new_t_id > t_id:
        t_id = new_t_id

for i in tqdm(range(N)):
    edges.lookup(t_id, f"test_{i}")


start = time.time()
print(len(list(edges.iterate(t_id, "node"))))
elapsed = time.time() - start
print(elapsed)