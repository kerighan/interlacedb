from numpy.core.numeric import errstate
import mmh3
import numpy as np
from interlacedb.database import InterlaceDB


class HashTable:
    def _get_header_fields(self):
        return {}

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
        n_bloom_filters=10, bloom_seed=12, cache_len=0
    ):
        self.key = key
        self.p_init = p_init
        self.probe_factor = probe_factor
        self.growth_factor = growth_factor
        self.n_bloom_filters = n_bloom_filters
        self.bloom_seed = bloom_seed
       
        # cache management
        self.cache_len = cache_len

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
        
        if self.cache_len > 0:
            from lru import LRU
            self.cache = LRU(self.cache_len)

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
                self.find_lookup_position = self.find_lookup_position_filtered
        else:
            self._load_tables_id()
            self.p_last = np.max(np.nonzero(self.tables_id)) + self.p_init

            if self.n_bloom_filters > 0:
                self._bloom_id = self._db.header[self._bloom_id_key]
                self._load_bloom_filters()
                self.find_lookup_position = self.find_lookup_position_filtered

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

    def _get_range(self, p,):
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
            self._insert_in_bloom(p, key)
        if self.cache_len > 0:
            self.cache[key] = p, position

    def _insert_in_bloom(self, p, key):
        key_hash = self._hash(key, self.bloom_seed)
        bloom_p = self.bloom_filters[p - self.p_init]
        bloom_capacity = self._get_capacity(p) * self.n_bloom_filters
        bucket = key_hash % bloom_capacity
        self._bloom.set_value(bloom_p, bucket, 1)

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

    def find_lookup_position_filtered(self, key, key_hash):
        if self.cache_len > 0:
            p, position = self.cache.get(key, (None, None))
            if p is not None:
                return p, position

        bloom_hash = self._hash(key, self.bloom_seed)
        for p in range(self.p_last, self.p_init - 1, -1):
            bloom_p = self.bloom_filters[p - self.p_init]
            bloom_capacity = self._get_capacity(p) * self.n_bloom_filters
            bucket = bloom_hash % bloom_capacity
            if self._bloom.get_value(bloom_p, bucket) == 0:
                continue
            try:
                p, position = self.find_lookup_position_in_table(
                    key, key_hash, p)
                return p, position
            except KeyError:
                pass
        raise KeyError

    def find_lookup_position(self, key, key_hash):
        if self.cache_len > 0:
            p, position = self.cache.get(key, (None, None))
            if p is not None:
                return p, position

        for p in range(self.p_last, self.p_init - 1, -1):
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
        # remove from cache
        if self.cache_len > 0:
            if key in self.cache:
                del self.cache[key]

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


class FloatingLayerTable(HashTable):
    def __init__(
        self,
        dataset,
        key,
        growth_factor=2,
        probe_factor=.5,
        p_init=0,
        n_bloom_filters=10,
        bloom_seed=12,
        cache_len=100000
    ):
        self.dataset = dataset
        self.key = key
        self.probe_factor = probe_factor
        self.growth_factor = growth_factor
        self.p_init = p_init
        self.n_bloom_filters = n_bloom_filters
        self.bloom_seed = bloom_seed
        self.cache_len = cache_len

        self._group_name = f"{dataset.name}_FLT_table"
        self._bloom_filter_name = f"{dataset.name}_FLT_filter"
        
    def _initialize(self):
        self.table = self._db.create_group(
            self._group_name, self.dataset,
            _prev_table="uint64", _p="uint8", _bloom_filter="uint64")
        self.table._add_database_reference(self._db)
        # create bloom filters
        self._bloom = self._db.create_array(self._bloom_filter_name, "bool")

        if self.cache_len > 0:
            from lru import LRU
            self.cache = LRU(self.cache_len)

    def _get_capacity(self, p):
        capacity = (self.growth_factor**p) -1
        return max(capacity, 1)

    def _get_range(self, p, capacity):
        m = int(round(p * self.probe_factor * self.growth_factor))
        m = min(max(1, m), capacity)
        return range(m)

    def new_table(self, p=None, _prev=None):
        if p is None:
            p = self.p_init
        capacity = self._get_capacity(p)
        
        # allocate new table and new bloom filter
        table_id = self.table.new_block(capacity)
        bloom_id = self._bloom.new_block(
            capacity * self.n_bloom_filters)

        # fill data
        self.table[table_id, "_p"] = p
        self.table[table_id, "_bloom_filter"] = bloom_id
        if _prev is not None:
            self.table[table_id, "_prev_table"] = _prev
        else:
            _prev = 0
        if self.cache_len > 0:
            self.cache[table_id] = (_prev, p, bloom_id)
        return table_id
    
    def insert(self, table_id, data):
        key = data[self.key]
        _hash = self._hash(key)
        metadata = self._get_metadata(table_id)
        
        # get bloom hash
        _bloom_hash = self._hash(key, seed=self.bloom_seed)

        try:
            t_id, position, capacity, bloom_id, new = self._find_lookup_position(
                table_id, _hash, _bloom_hash, key, metadata)
        except KeyError:
            t_id, position, capacity, bloom_id, new = self._find_insert_position(
                table_id, _hash, key, metadata)
        
        self.table[t_id, position] = data
        if True:
            self._insert_in_bloom(bloom_id, capacity, _bloom_hash)
        return t_id

    def lookup(self, table_id, key):
        _hash = self._hash(key)
        _bloom_hash = self._hash(key, seed=self.bloom_seed)
        metadata = self._get_metadata(table_id)
        t_id, position, _, _, _ = self._find_lookup_position(
            table_id, _hash, _bloom_hash, key, metadata, verbose=True)
        return self.table[t_id, position]

    def _insert_in_bloom(self, bloom_id, capacity, _bloom_hash):
        bloom_capacity = capacity * self.n_bloom_filters
        bucket = _bloom_hash % bloom_capacity
        self._bloom.set_value(bloom_id, bucket, 1)
    
    def _lookup_in_bloom(self, bloom_id, capacity, _bloom_hash):
        bloom_capacity = capacity * self.n_bloom_filters
        bucket = _bloom_hash % bloom_capacity
        return self._bloom.get_value(bloom_id, bucket)
    
    def _get_metadata(self, table_id):
        if self.cache_len > 0:
            metadata = self.cache.get(table_id)
            if metadata is None:
                m = self.table[table_id]
                _prev, p, bloom_id = m["_prev_table"], m["_p"], m["_bloom_filter"]
                self.cache[table_id] = (_prev, p, bloom_id)
                return (_prev, p, bloom_id)
            else:
                return metadata
        
        m = self.table[table_id]
        metadata = m["_prev_table"], m["_p"], m["_bloom_filter"]
        return metadata

    def _find_lookup_position(self, table_id, _hash, _bloom_hash, key, metadata, verbose=False):
        _get_status = self.table.status
        _get_value = self.table.get_data_value

        _prev, p_max, bloom_id = metadata
        for p in range(p_max, self.p_init-1, -1):
            capacity = self._get_capacity(p)
            # check if value is in bloom filter
            bloom_value = self._lookup_in_bloom(
                bloom_id, capacity, _bloom_hash)
            if bloom_value != 0:
                bucket = _hash % capacity
                for i in self._get_range(p, capacity):
                    position = (bucket + i) % capacity
                    status = _get_status(table_id, position)
                    if status == 0:
                        break
                    elif status == 1:
                        current_key = _get_value(table_id, position, self.key)
                        if current_key == key:
                            return (table_id,
                                    position,
                                    capacity,
                                    bloom_id,
                                    False)
            table_id = _prev
            _prev, _, bloom_id = self._get_metadata(table_id)
            
        raise KeyError

    def _find_insert_position(self, table_id, _hash, key, metadata):
        _get_status = self.table.status

        _prev, p_max, bloom_id = metadata
        table_id_max = int(table_id)
        for p in range(p_max, self.p_init-1, -1):
            capacity = self._get_capacity(p)
            bucket = _hash % capacity
            for i in self._get_range(p, capacity):
                position = (bucket + i) % capacity
                status = _get_status(table_id, position)
                if status <= 0:  # slot is free or deleted
                    return (table_id,
                            position,
                            capacity,
                            bloom_id,
                            True)
                else:
                    # no need to check for equality: 
                    # it was already been done in lookup phase
                    continue

            if _prev == 0:
                break
            table_id = _prev
            _prev, _, bloom_id = self._get_metadata(table_id)

        # create a new table and fill data
        new_table_id = self.new_table(p=p_max+1, _prev=table_id_max)
        metadata = self._get_metadata(new_table_id)
        return self._find_insert_position(new_table_id, _hash, key, metadata)

    def iterate(self, table_id, field=None):
        metadata = self._get_metadata(table_id)
        if metadata is None:
            metadata = self.table[table_id]
            p = metadata["_p"]
            _prev = metadata["_prev_table"]
            bloom_id = metadata["_bloom_filter"]
            self.cache[table_id] = (_prev, p, bloom_id)
        else:
            _prev, p, _ = metadata

        while True:
            capacity = self._get_capacity(p)
            if field is None:
                items = self.table[table_id, :capacity]
            else:
                items = self.table[table_id, :capacity, field]
            for it in items:
                if it is not None:
                    yield it
            table_id = _prev
            if table_id == 0:
                break
            meta = self.table[table_id]
            _prev = meta["_prev_table"]
            p = meta["_p"]
