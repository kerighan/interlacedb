import mmh3


class HashTree:
    def __init__(self, row, p_init=12):
        self.p_init = p_init
        self.offset = 6
        self.row = row
        row_name = self.row.name
        self.row_name = row_name
        self._root_name = f"_root_{row_name}"
        self._p_name = f"_p_{row_name}"
        self._n_name = f"_n_{row_name}"
        self._meta_start_name = f"_meta_start_{row_name}"

        self.db = row.db
        self.table = self.db.create_dataset(
            f"_table_{row_name}",
            _taken="bool_",
            _hash="uint64",
            _distance="uint8",
            **self.row.field_to_dtype)

    def _get_hash(self, key):
        return mmh3.hash(key, signed=False)

    def _get_header(self):
        fields = {
            self._p_name: "uint8",
            self._n_name: "uint64"
        }
        # add table meta data
        for i in range(self.p_init, 64):
            fields[f"_meta_{i}_{self.row_name}"] = "uint64"
        return fields

    def _finalize(self):
        self.header = self.db.header
        self.p = self.p_init
        self.capacity = 2**self.p

        self.start, _ = self.table.allocate(self.capacity)
        self.header[self._p_name] = self.p
        self.header[f"_meta_{self.p}_{self.row_name}"] = self.start

        self.starting_points = [0] * 64
        self.starting_points[self.p] = self.start

    def _size_up(self):
        self.p += 1
        self.capacity = 2**self.p

        self.start, _ = self.table.allocate(self.capacity)
        self.header[self._p_name] = self.p
        self.header[f"_meta_{self.p}_{self.row_name}"] = self.start

        self.starting_points = [0] * 64
        self.starting_points[self.p] = self.start
        for i in range(self.p_init, 64):
            self.starting_points[i] = self.header[f"_meta_{i}_{self.row_name}"]
    
    def lookup(self, key):
        hash = self._get_hash(key)
        for p in range(self.p, self.p_init-1, -1):
            try:
                res = self.lookup_at_p(hash, p)
                return res
            except KeyError:
                continue
        raise KeyError

    def lookup_at_p(self, hash, p):
        if p == self.p:
            start = self.start
        else:
            start = self.header[f"_meta_{p}_{self.row_name}"]
        
        capacity = 2**p
        bucket = hash % capacity
        table_at = self.table.at
        for _ in range(p - self.offset):
            existing_taken = table_at[start, bucket, "_taken"]
            # slot is free
            if not existing_taken:
                raise KeyError

            # element found
            existing_hash = table_at[start, bucket, "_hash"]
            if existing_hash == hash:
                res = table_at[start, bucket]
                del res["_taken"]
                del res["_hash"]
                del res["_distance"]
                return res

            bucket = (bucket + 1) % capacity
        raise KeyError

    def insert(self, key, value):
        hash = self._get_hash(key)
        self.insert_at_p(hash, value, self.p_init)
    
    def insert_at_p(self, hash, value, p):
        value["_taken"] = True
        value["_hash"] = hash

        bucket = hash % self.capacity
        table_at = self.table.at
        start = self.start
        distance = 0
        while True:
            existing_taken = table_at[start, bucket, "_taken"]
            # slot is free
            if not existing_taken:
                value["_distance"] = distance
                value["_hash"] = hash
                table_at[start, bucket] = value
                return

            # element found
            existing_hash = table_at[start, bucket, "_hash"]
            if existing_hash == hash:
                return
            
            # robin hood
            existing_distance = table_at[start, bucket, "_distance"]
            if existing_distance < distance:
                existing_value = table_at[start, bucket]
                # replace existing value at bucket with value
                value["_distance"] = distance
                table_at[start, bucket] = value
                # swap values
                hash = existing_value["_hash"]
                distance = existing_value["_distance"]
                value = existing_value

            bucket = (bucket + 1) % self.capacity
            distance += 1

            if distance >= p - self.offset:
                if p == self.p:
                    self._size_up()
                    self.insert_at_p(hash, value, self.p)
                else:
                    self.insert_at_p(hash, value, p + 1)
                return


class HashTable:
    def __init__(self, db, row, p_init=12):
        self.p_init = p_init
        self.offset = 6
        self.row = row
        row_name = self.row.name
        self.row_name = row_name
        self._root_name = f"_root_{row_name}"
        self._p_name = f"_p_{row_name}"
        self._n_name = f"_n_{row_name}"
        self._meta_start_name = f"_meta_start_{row_name}"

        self.db = db
        self.table = self.db.create_dataset(
            f"_table_{row_name}",
            _taken="bool_",
            _hash="uint64",
            _distance="uint8",
            **self.row.field_to_dtype)

    def _get_hash(self, key):
        return mmh3.hash(key, signed=False)

    def _get_header(self):
        fields = {
            self._p_name: "uint8",
            self._n_name: "uint64"
        }
        # add table meta data
        for i in range(self.p_init, 64):
            fields[f"_meta_{i}_{self.row_name}"] = "uint64"
        return fields
