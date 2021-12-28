from random import randint

from tqdm import tqdm

from interlacedb import InterlaceDB

with InterlaceDB("test.db", flag="n") as db:
    array = db.create_array("array", "bool")


block_id = array.new_block(1000000)

N = 500000
for _ in tqdm(range(N)):
    i = randint(0, 100000-1)
    array[block_id, i] = True

for _ in tqdm(range(N)):
    i = randint(0, 100000-1)
    try:
        array[block_id, i]
    except IndexError:
        print(i)
