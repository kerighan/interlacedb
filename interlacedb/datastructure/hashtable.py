import mmh3
import numpy as np
from interlacedb.database import InterlaceDB


class HashTable:
    def _remove_database_reference(self):
        if hasattr(self, "_db"):
            del self._db

    def _add_database_reference(self, db):
        self._db = db

    def _hash(self, key, seed=0):
        if not isinstance(key, str):
            key = str(key)
        return mmh3.hash(key, seed=seed, signed=False)

    def __contains__(self, key):
        return self.contains(key)

    def __getitem__(self, key):
        return self.lookup(key)

    def __setitem__(self, key, data):
        data[self.key] = key
        self.insert(data)

    def __delitem__(self, key):
        self.delete(key)


class LayerTable(HashTable):
    def __init__(
        self, dataset, key, growth_factor=2, p_init=10, probe_factor=.5,
        n_bloom_filters=10, bloom_seed=12
    ):
        self.key = key
        self.p_init = p_init
        self.probe_factor = probe_factor
        self.growth_factor = growth_factor
        self.n_bloom_filters = n_bloom_filters
        self.bloom_seed = bloom_seed

        self.dataset = dataset
        self.dstruct_name = f"{dataset.name}_LHT"
        self.tables_id_key = f"{self.dstruct_name}_tables_id"
        self._block_id_key = f"{self.dstruct_name}_block_id"
        self._bloom_id_key = f"{self.dstruct_name}_bloom_id"
        self._bloom_filter_key = f"{self.dstruct_name}_bloom_filter"

    def _get_header_fields(self):
        return {
            f"{self._block_id_key}": "uint64",
            f"{self._bloom_id_key}": "uint64",
        }

    def _initialize(self):
        self._block_id = self._db.header[self._block_id_key]
        self._positions = self._db.create_array(self.tables_id_key, "uint64")
        self._bloom = self._db.create_array(self._bloom_filter_key, "bool")

        if self._block_id == 0:
            # create array of hashtables positions
            self._block_id = self._positions.new_block(32)
            self._db.header[self._block_id_key] = self._block_id
            # allocate first table
            capacity = self._get_capacity(self.p_init)
            table_id = self.dataset.new_block(capacity)
            self._positions.set_value(self._block_id, 0, table_id)
            self.p_last = self.p_init
            self._load_tables_id()

            if self.n_bloom_filters > 0:
                # create array of bloom filters positions
                self._bloom_id = self._positions.new_block(32)
                self._db.header[self._bloom_id_key] = self._bloom_id
                filter_id = self._bloom.new_block(
                    capacity * self.n_bloom_filters)
                self._positions.set_value(self._bloom_id, 0, filter_id)
                self._load_bloom_filters()
        else:
            self._load_tables_id()
            self.p_last = np.max(np.nonzero(self.tables_id)) + self.p_init

            if self.n_bloom_filters > 0:
                self._bloom_id = self._db.header[self._bloom_id_key]
                self._load_bloom_filters()

        self.get = self.dataset.get
        self.exists = self.dataset.exists
        self.status = self.dataset.status
        self.get_value = self.dataset.get_value

    def _save_tables_id(self, index, table_id):
        self._positions.set_value(self._block_id, index, table_id)

    def _load_tables_id(self):
        self.tables_id = list(
            self._positions.get_values(self._block_id, 0, 32))

    def _load_bloom_filters(self):
        self.bloom_filters = list(
            self._positions.get_values(self._bloom_id, 0, 32))

    def _get_range(self, p):
        return range(int(round(p * self.probe_factor * self.growth_factor)))

    def _get_capacity(self, p):
        return self.growth_factor**p

    def _create_new_hashtable(self):
        self.p_last += 1
        capacity = self._get_capacity(self.p_last)
        table_id = self.dataset.new_block(capacity)
        index = self.p_last - self.p_init
        self.tables_id[index] = table_id
        self._save_tables_id(index, table_id)

        if self.n_bloom_filters > 0:
            filter_id = self._bloom.new_block(
                capacity * self.n_bloom_filters)
            self._positions.set_value(self._bloom_id, index, filter_id)
            self.bloom_filters[index] = filter_id

    def insert(self, data):
        key = data[self.key]
        key_hash = self._hash(key)

        try:
            p, position = self.find_lookup_position(key, key_hash)
        except KeyError:
            p, position = self.find_insert_position(key, key_hash)
        table_id = self.tables_id[p - self.p_init]

        self.dataset.set(table_id, position, data)
        if self.n_bloom_filters > 0:
            self.insert_in_bloom(p, key)

    def insert_in_bloom(self, p, key):
        key_hash = self._hash(key, self.bloom_seed)
        bloom_p = self.bloom_filters[p - self.p_init]
        bloom_capacity = self._get_capacity(p) * self.n_bloom_filters
        bucket = key_hash % bloom_capacity
        self._bloom.set_value(bloom_p, bucket, 1)

    def lookup_in_bloom(self, key):
        key_hash = self._hash(key, self.bloom_seed)
        res = []
        for p in range(self.p_last, self.p_init - 1, -1):
            bloom_p = self.bloom_filters[p - self.p_init]
            bloom_capacity = self._get_capacity(p) * self.n_bloom_filters
            bucket = key_hash % bloom_capacity
            if self._bloom.get_value(bloom_p, bucket):
                res.append(p)
        return res

    def contains(self, key, key_hash=None):
        if key_hash is None:
            key_hash = self._hash(key)
        try:
            self.find_lookup_position(key, key_hash)
            return True
        except KeyError:
            return False

    def find_insert_position(self, key, key_hash):
        for p in range(self.p_last, self.p_init - 1, -1):
            try:
                p, position = self.find_insert_position_in_table(
                    key, key_hash, p)
                return p, position
            except KeyError:
                continue

        self._create_new_hashtable()
        p, position = self.find_insert_position_in_table(
            key, key_hash, self.p_last)
        return p, position

    def find_insert_position_in_table(self, key, key_hash, p):
        key_name = self.key

        capacity = self._get_capacity(p)
        bucket = key_hash % capacity
        table_id = self.tables_id[p - self.p_init]
        for i in self._get_range(p):
            position = (bucket + i) % capacity
            status = self.status(table_id, position)
            if status == 1:
                key_current = self.get_value(
                    table_id, position, key_name)
                if key_current != key:
                    continue
                return p, position
            else:
                return p, position
        raise KeyError

    def lookup(self, key):
        key_hash = self._hash(key)
        p, position = self.find_lookup_position(key, key_hash)
        table_id = self.tables_id[p - self.p_init]
        return self.get(table_id, position)

    def find_lookup_position(self, key, key_hash):
        if self.n_bloom_filters == 0:
            candidates = range(self.p_last, self.p_init - 1, -1)
        else:
            candidates = self.lookup_in_bloom(key)

        for p in candidates:
            try:
                p, position = self.find_lookup_position_in_table(
                    key, key_hash, p)
                return p, position
            except KeyError:
                pass
        raise KeyError

    def find_lookup_position_in_table(self, key, key_hash, p):
        key_name = self.key

        capacity = self._get_capacity(p)
        bucket = key_hash % capacity
        table_id = self.tables_id[p - self.p_init]
        for i in self._get_range(p):
            position = (bucket + i) % capacity
            status = self.status(table_id, position)
            if status == 1:
                key_current = self.get_value(table_id, position, key_name)
                if key_current != key:
                    continue
                return p, position
            elif status == -1:
                continue
            else:
                raise KeyError
        raise KeyError

    def delete(self, key):
        key_hash = self._hash(key)
        p, position = self.find_lookup_position(key, key_hash)
        table_id = self.tables_id[p - self.p_init]
        self.dataset.delete(table_id, position)

    def __iter__(self):
        for p in range(self.p_last - self.p_init + 1):
            table_id = self.tables_id[p]
            capacity = self._get_capacity(p + self.p_init)
            for i in range(capacity):
                if self.exists(table_id, i):
                    yield self.get(table_id, i)


class Dict:
    def __init__(self, filename, size=1024, **kwargs):
        from numpy import log2
        p_init = int(round(log2(size)))

        import os
        if not os.path.exists(filename):
            with InterlaceDB(filename, **kwargs) as db:
                dset = db.create_dataset("dset", key="uint64", value="blob")
                dstruct = db.create_datastructure(
                    "dstruct",
                    LayerTable(
                        dset, "key",
                        n_bloom_filters=20,
                        p_init=p_init,
                        probe_factor=.3))
        else:
            db = InterlaceDB(filename, **kwargs)
            dstruct = db.datastructures["dstruct"]
        self.dstruct = dstruct

    def _hash(self, key):
        from cityhash import CityHash64
        res = CityHash64(key)
        if res != 0:
            return res
        return 1

    def insert(self, key, value):
        key_hash = self._hash(key)
        self.dstruct[key_hash] = {"value": (key, value)}

    def get(self, key, res=None):
        key_hash = self._hash(key)
        try:
            return self.dstruct[key_hash]["value"][1]
        except KeyError:
            return res

    def __setitem__(self, key, value):
        self.insert(key, value)

    def __getitem__(self, key):
        res = self.get(key)
        if res is not None:
            return res
        raise KeyError

    def __contains__(self, key):
        key_hash = self._hash(key)
        return key_hash in self.dstruct

    def __iter__(self):
        for data in self.dstruct:
            yield data["value"]


class FracTable(HashTable):
    def __init__(
        self, dataset, key,
        p_min=2, p_init=9
    ):
        self.key = key
        self.p_init = p_init
        self.p_min = p_min

        self.dataset = dataset
        self.dstruct_name = f"{dataset.name}_FT"
        self._capsule_start_key = f"{self.dstruct_name}_capsule_start"
        self._capsule_key = f"{self.dstruct_name}_capsule"

    def _get_header_fields(self):
        return {
            f"{self._capsule_start_key}": "uint64"
        }

    def _initialize(self):
        self._capsule_start = self._db.header[self._capsule_start_key]
        self._capsule = self._db.create_array(self._capsule_key, "uint64")

        if self._capsule_start == 0:
            size_init = self._get_capacity(0)
            self._capsule_start = self._capsule.new_block(3 * size_init)
            self._db.header[self._capsule_start_key] = self._capsule_start
        else:
            pass

        self.get = self.dataset.get
        self.exists = self.dataset.exists
        self.status = self.dataset.status
        self.get_value = self.dataset.get_value

    def insert(self, data):
        key = data[self.key]
        key_hash = self._hash(key)
        caps_pos, position, empty = self._find_position(key_hash, key)
        self._insert_new(caps_pos, position, key_hash, data)

    def lookup(self, key):
        key_hash = self._hash(key)
        caps_pos, position, empty = self._find_position(key_hash, key)
        if empty:
            raise KeyError
        return self._lookup(caps_pos, position)

    def _insert_new(self, caps_pos, position, key_hash, data):
        data_id = self.dataset.append(**data)
        self._capsule[caps_pos, position] = key_hash
        self._capsule[caps_pos, position + 1] = data_id

    def _lookup(self, caps_pos, position):
        data_id = self._capsule[caps_pos, position + 1]
        return self.dataset[data_id, 0]

    def contains(self, key):
        key_hash = self._hash(key)
        _, _, empty = self._find_position(key_hash, key)
        return not empty

    def _get_capacity(self, depth):
        # if depth == 0:
        #     capacity = self.size_init
        # else:
        #     capacity = self.growth_factor
        capacity = max(2**self.p_min,
                       2**(self.p_init - depth))
        return capacity - 1

    def _find_position(self, key_hash, key):
        depth = 0
        caps_pos = self._capsule_start
        while True:
            capacity = self._get_capacity(depth)

            # 3 because: hash, item_position, next_capsule
            position = 3 * (key_hash % capacity)

            current_hash = self._capsule[caps_pos, position]
            if current_hash == 0:  # capsule free
                return caps_pos, position, True
            elif current_hash == key_hash:
                return caps_pos, position, False
            else:
                next_capsule_pos = self._capsule[caps_pos, position + 2]
                if next_capsule_pos == 0:
                    next_capacity = self._get_capacity(depth + 1)
                    next_capsule_pos = self._capsule.new_block(
                        3 * next_capacity)
                    self._capsule[caps_pos, position + 2] = next_capsule_pos
                caps_pos = next_capsule_pos
                depth += 1
