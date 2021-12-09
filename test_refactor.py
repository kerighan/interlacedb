from interlacedb import InterlaceDB
import os

# with InterlaceDB("test.db", flag="n") as db:
#     db.create_header(n_nodes="uint64")
#     dset = db.create_dataset("nodes", key="U20", value="blob")

# print(db.datasets)
db = InterlaceDB("test.db")
# db.index = 55
print(db.index)

# print(dset)
# print(db.table_start)
# dset.allocate(10)
