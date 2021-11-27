from numpy import array, dtype, frombuffer, int8, integer, uint32, str_


blob_dt = dtype([("blob", uint32)])


class Row:
    def __init__(self, identifier, db, name, dtypes, offset=0):
        self.identifier = identifier
        self.db = db
        self.name = name
        self.dtypes = dtypes
        self.offset = offset
        self.prefix = int8(identifier).tobytes()
        self.prefix_size = len(self.prefix)

        # add methods from db
        self.from_unit = db.from_unit
        self.to_unit = db.to_unit
        self.read_at = db.read_at
        self.write_at = db.write_at
        self.db_append = db.append
        self.db_append_blob = db.append_blob
        self.db_get_blob = db.get_blob

        self._build()

    def _build(self):
        field_to_dtype = {}
        self.size_align_dtype = {}
        self.string_fields = set()
        self.blob_fields = set()
        cum_pos = self.prefix_size
        for key, dt in self.dtypes:
            item_size = dt.itemsize
            field_to_dtype[key] = dt
            self.size_align_dtype[key] = (item_size, cum_pos, dt)

            if dt != blob_dt:  # item is regular field
                cum_pos += item_size
                if dt.type is str_:
                    self.string_fields.add(key)
            else:  # item is blob
                cum_pos += 4
                self.blob_fields.add(key)
        self.len = cum_pos
        self.field_to_dtype = field_to_dtype
        self.fields = list(self.field_to_dtype.keys())
        self.has_blob = len(self.blob_fields) != 0
    
    def _finalize(self):
        self.size = self.to_unit(self.len)
        self.at = At(self)

    def __repr__(self):
        txt = []
        for key, dt in self.dtypes:
            txt.append(f"{key}={dt}")
        txt = ", ".join(txt)
        return self.name.capitalize() + "(" + txt + ")"

    def __len__(self):
        return self.len

    def encode(self, **data):
        for f in self.string_fields:
            if f not in data:
                data[f] = ""
        if self.has_blob:
            for f in self.blob_fields:
                if f not in data:
                    data[f] = 0
                    continue
                data[f]
        res = tuple(data.get(key, 0) for key in self.fields)
        return array(res, dtype=self.dtypes)

    def to_bytes(self, **data):
        if self.has_blob:
            self.parse_blob(data)
        arr = self.encode(**data)
        return self.prefix + arr.tobytes()

    def allocate(self, size):
        row_size = self.to_unit(self.len)
        return self.db.allocate(row_size * size)

    def append(self, **data):
        return self.db_append(self.to_bytes(**data))

    def parse_blob(self, data):
        for field in self.blob_fields:
            if field not in data:
                continue
            data[field] = self.db_append_blob(data[field])

    def _parse(self, res):
        res = frombuffer(res, dtype=self.dtypes)[0]
        if not self.has_blob:
            return dict(zip(self.fields, res))

        res = dict(zip(self.fields, res))
        for field in self.blob_fields:
            blob_id = res[field][0]
            if blob_id == 0:
                del res[field]
                continue
            res[field] = self.db_get_blob(blob_id)
        return res

    def _get_item_inplace(self, field):
        size, align, dt = self.size_align_dtype[field]

        data_bytes = self.read_at(
            self.offset + align, size)
        res = frombuffer(data_bytes, dtype=dt)[0]
        return res

    def _get_row(self, index):
        # check row is indeed of good type
        byte_index = self.from_unit(index)
        data_bytes = self.read_at(byte_index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier != self.identifier:
            raise KeyError
        byte_index += self.prefix_size
        data_bytes = self.read_at(byte_index, self.len - self.prefix_size)
        return self._parse(data_bytes)

    def _get_item(self, index, field):
        size, align, dt = self.size_align_dtype[field]

        byte_index = self.from_unit(index)
        data_bytes = self.read_at(
            byte_index + align, size)
        res = frombuffer(data_bytes, dtype=dt)[0]
        return res
    
    def _get_items(self, start, stop):
        n_items = stop - start

        byte_index = self.from_unit(start)
        data_bytes = self.read_at(byte_index, self.len * n_items)
        res = frombuffer(data_bytes, dtype=[("prefix", int8)] + self.dtypes)
        data = [{key: item[key] for key in self.fields} for item in res]
        return data

    def _set_item_inplace(self, field, value):
        _, align, dt = self.size_align_dtype[field]
        data = array(value, dtype=dt)
        self.write_at(self.offset + align, data)

    def _set_row(self, index, data):
        data_bytes = self.to_bytes(**data)
        byte_index = self.from_unit(index)
        self.write_at(byte_index, data_bytes)

    def _set_item(self, index, field, value):
        _, align, dt = self.size_align_dtype[field]
        data = array(value, dtype=dt)
        byte_index = self.from_unit(index)
        self.write_at(byte_index + align, data)
    
    def _set_rows(self, start, rows):
        byte_index = self.from_unit(start)
        data_bytes = b''.join(self.to_bytes(**row) for row in rows)
        self.write_at(byte_index, data_bytes)
    
    def __delitem__(self, start):
        byte_index = self.from_unit(start)
        data_bytes = self.read_at(byte_index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        assert identifier == self.identifier
        self.write_at(byte_index, int8(-identifier).tobytes())

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._get_item_inplace(key)
        elif isinstance(key, (int, integer)):
            return self._get_row(key)
        elif isinstance(key, tuple):
            index, field = key
            return self._get_item(index, field)
        elif isinstance(key, slice):
            start = key.start or 0
            stop = key.stop
            assert stop is not None
            return self._get_items(start, stop)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            return self._set_item_inplace(key, value)
        elif isinstance(key, (int, integer)):
            return self._set_row(key, value)
        elif isinstance(key, tuple):
            index, field = key
            return self._set_item(index, field, value)
        elif isinstance(key, slice):
            start = key.start
            assert start is not None
            return self._set_rows(start, value)
    
    def __iter__(self):
        for data in self.db.iter_dataset(self):
            yield data
    
    def slice(self, start, end):
        for data in self.db.iter_dataset(self, start, end):
            yield data


class At:
    def __init__(self, row):
        self.set = row.__setitem__
        self.get = row.__getitem__
        self.delete = row.__delitem__
        self.size = row.size

    def __getitem__(self, key):
        if len(key) == 2:
            index = int(key[0] + key[1] * self.size)
            return self.get(index)
        elif len(key) == 3:
            index = int(key[0] + key[1] * self.size)
            return self.get((index, key[2]))

    def __setitem__(self, key, value):
        if len(key) == 2:
            index = int(key[0] + key[1] * self.size)
            self.set(index, value)
        elif len(key) == 3:
            index = int(key[0] + key[1] * self.size)
            self.set((index, key[2]), value)
    
    def __delitem__(self, key):
        index = key[0] + key[1] * self.size
        self.delete(index)
