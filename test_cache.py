from cachetools import LRUCache
from tqdm import tqdm
from lru import LRU


# cache = LRUCache(10000)
cache = LRU(10000000)

N = 10000000
for i in tqdm(range(N)):
    cache[f"test_{i}"] = i
print(len(cache))
print(cache["test_9999999"])

for i in tqdm(range(N)):
    cache.get(f"test_{i}")
