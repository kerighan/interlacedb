from interlacedb import InterlaceDB
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

if os.path.exists("test.db"):
    os.remove("test.db")

with InterlaceDB("test.db") as db:
    key = db.create_dataset("key", key="U20")


start, end = key.fill(5000)
for i in tqdm(range(5000)):
    key.at[start, i] = dict(key=f"key_{i}")

df = pd.DataFrame(key.at[start:end])
print(df)


# for data in tqdm(key, total=5000):
#     pass

# node.append(key=1, test=[[1, 3], [1, 3]])
# node.append(key=2, test=[[1, 3], [1, 3]])

# start, end = node.fill(50000)
# for i in tqdm(range(50000)):
#     node.at[start, i] = {"key": i}

# for i in tqdm(range(50000)):
#     node.at[start, i]
