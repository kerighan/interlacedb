import mmh3
import numpy as np


class LayeredHashTable:
    def __init__(self, dataset, key, branching_factor=2, p_init=10, probe_factor=.5):
        self.key = key
        self.p_init = p_init
        self.probe_factor = probe_factor
        self.branching_factor = branching_factor

        self.dataset = dataset
        self.dstruct_name = f"{dataset.name}_LHT"
        self.tables_id_key = f"{self.dstruct_name}_tables_id"
        self._block_id_key = f"{self.dstruct_name}_block_id"

    def _get_header_fields(self):
        return {f"{self._block_id_key}": "uint64"}

    def _initialize(self):
        self._block_id = self._db.header[self._block_id_key]
        self._arr = self._db.create_array(self.tables_id_key, "uint64")
        if self._block_id == 0:
            # create array
            self._block_id = self._arr.new_block(32)
            self._db.header[self._block_id_key] = self._block_id

            table_id = self.dataset.new_block(self._get_capacity(self.p_init))
            self._arr.set_value(self._block_id, 0, table_id)
            self.p_last = self.p_init
            self._load_tables_id()
        else:
            self._load_tables_id()
            self.p_last = np.max(np.nonzero(self.tables_id)) + self.p_init

    def _save_tables_id(self, index, table_id):
        self._arr.set_value(self._block_id, index, table_id)

    def _load_tables_id(self):
        self.tables_id = list(self._arr.get_values(self._block_id, 0, 32))

    def _remove_database_reference(self):
        if hasattr(self, "_db"):
            del self._db

    def _add_database_reference(self, db):
        self._db = db

    def _hash(self, key):
        if not isinstance(key, str):
            key = str(key)
        return mmh3.hash(key, signed=False)

    def _get_range(self, p):
        return range(int(round(p * self.probe_factor * self.branching_factor)))

    def _get_capacity(self, p):
        return self.branching_factor**p

    def _create_new_hashtable(self):
        self.p_last += 1
        capacity = self._get_capacity(self.p_last)
        table_id = self.dataset.new_block(capacity)
        index = self.p_last - self.p_init
        self.tables_id[index] = table_id
        self._save_tables_id(index, table_id)

    def insert(self, data):
        key = data[self.key]
        key_hash = self._hash(key)

        p, position = self.find_insert_position(key, key_hash)
        table_id = self.tables_id[p - self.p_init]

        self.dataset.set(table_id, position, data)

    def find_insert_position(self, key, key_hash):
        for p in range(self.p_last, self.p_init - 1, -1):
            try:
                p, position = self.find_insert_position_in_table(
                    key, key_hash, p)
                return p, position
            except KeyError:
                pass

        self._create_new_hashtable()
        p, position = self.find_insert_position_in_table(
            key, key_hash, self.p_last)
        return p, position

    def find_insert_position_in_table(self, key, key_hash, p):
        exists = self.dataset.exists
        get_value = self.dataset.get_value
        key_name = self.key

        capacity = self._get_capacity(p)
        bucket = key_hash % capacity
        table_id = self.tables_id[p - self.p_init]
        for i in self._get_range(p):
            position = (bucket + i) % capacity
            if exists(table_id, position):
                key_current = get_value(table_id, position, key_name)
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
        return self.dataset.get(table_id, position)

    def find_lookup_position(self, key, key_hash):
        for p in range(self.p_last, self.p_init - 1, -1):
            try:
                p, position = self.find_lookup_position_in_table(
                    key, key_hash, p)
                return p, position
            except KeyError:
                pass
        raise ValueError

    def find_lookup_position_in_table(self, key, key_hash, p):
        exists = self.dataset.exists
        get_value = self.dataset.get_value
        key_name = self.key

        capacity = self._get_capacity(p)
        bucket = key_hash % capacity
        table_id = self.tables_id[p - self.p_init]
        for i in self._get_range(p):
            position = (bucket + i) % capacity
            if exists(table_id, position):
                key_current = get_value(table_id, position, key_name)
                if key_current != key:
                    continue
                return p, position
            else:
                raise KeyError
        raise KeyError
