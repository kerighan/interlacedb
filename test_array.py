from random import randint

from numpy.core.shape_base import block
from tqdm import tqdm

from interlacedb import InterlaceDB

with InterlaceDB("test.db", flag="n") as db:
    array = db.create_array("array", "bool")


block_id = array.new_block(100000)

N = 50000
for _ in tqdm(range(N)):
    i = randint(0, 100000-1)
    array[block_id, i] = True

for _ in tqdm(range(N)):
    i = randint(0, 100000-1)
    i = 100000-1
    try:
        array[block_id, i]
    except IndexError:
        print(i)

print(array[block_id, 500:5000])
