from sqlitedict import SqliteDict
from tqdm import tqdm
import random

db = SqliteDict("test.sqlite", autocommit=True)

N = 100000
for i in tqdm(range(N)):
    db[f"test_{i}"] = i

for _ in tqdm(range(N)):
    i = random.randint(0, N-1)
    db[f"test_{i}"]
