from numpy import array, dtype, frombuffer, int8, integer, str_, uint32

blob_dt = dtype([("blob", uint32)])
PREFIX_DTYPE = int8


class Dataset:
    def __init__(self, identifier, db, name, dtypes, offset=0):
        self.name = name
        self._identifier = identifier
        self._dtypes = dtypes
        self._len = 0
        self._prefix = PREFIX_DTYPE(identifier).tobytes()
        self._prefix_size = len(self._prefix)
        self._offset = 0

        # add methods from db
        self._db = db
        self._read_at = db._read_at
        self._write_at = db._write_at
        self._db_append = db._append
        self._db_allocate = db._allocate
        # self._db_append_blob = db.append_blob
        # self._db_get_blob = db.get_blob

        self._compile()

    def _remove_database_reference(self):
        del self._db
        del self._read_at
        del self._write_at
        del self._db_append
        del self._db_allocate
        # del self._db_append_blob
        # del self._db_get_blob

    def _add_database_reference(self, db):
        self._db = db
        self._read_at = db._read_at
        self._write_at = db._write_at
        self._db_append = db._append
        self._db_allocate = db._allocate
        # self._db_append_blob = db.append_blob
        # self._db_get_blob = db.get_blob

    def _compile(self):
        self._blob_fields = set()
        self._string_fields = set()
        self._field = {}

        position = self._prefix_size
        for key, dt in self._dtypes:
            dt_size = dt.itemsize
            self._field[key] = (dt.itemsize, position, dt)
            if dt != blob_dt:
                if dt.type is str_:  # faster that way
                    self._string_fields.add(key)
            else:
                self._blob_fields.add(key)
            position += dt_size
        self._len = position
        self._has_blob = len(self._blob_fields) != 0

    # =========================================================================
    # encoding and decoding functions
    # =========================================================================

    def _to_numpy(self, data):
        for f in self._string_fields:
            if f not in data:
                data[f] = ""
        if self._has_blob:
            for f in self._blob_fields:
                if f not in data:
                    data[f] = 0
                    continue
        res = tuple(data.get(key, 0) for key in self._field)
        return array(res, dtype=self._dtypes)

    def _to_bytes(self, data):
        if self._has_blob:
            # self.parse_blob(data)
            pass
        arr = self._to_numpy(data)
        return self._prefix + arr.tobytes()

    def _parse(self, res):
        res = frombuffer(res, dtype=self._dtypes)[0]
        if not self._has_blob:
            return dict(zip(self._field, res))

        res = dict(zip(self._field, res))
        for field in self._blob_fields:
            blob_id = res[field][0]
            if blob_id == 0:
                del res[field]
                continue
            # res[field] = self._db_get_blob(blob_id)
        return res

    # =========================================================================
    # overloading functions
    # =========================================================================

    def __repr__(self):
        txt = []
        for key, dt in self._dtypes:
            txt.append(f"{key}={dt}")
        txt = ", ".join(txt)
        return self.name.capitalize() + "(" + txt + ")"

    def __len__(self):
        return self._len

    # =========================================================================
    # setters and getters
    # =========================================================================

    def _set_field_no_index(self, key, value):
        _, align, dt = self._field[key]
        data = array(value, dtype=dt)
        self._write_at(self._offset + align, data)

    def _get_field_no_index(self, key):
        dt_size, align, dt = self._field[key]
        data_bytes = self._read_at(self._offset + align, dt_size)
        res = frombuffer(data_bytes, dtype=dt)[0]
        return res

    # def _set_item(self, byte_index, key, value):
    #     _, align, dt = self._field[key]
    #     data = array(value, dtype=dt)
    #     self._write_at(byte_index + align, data)

    def _get_index_from(self, block_index, row_index):
        return block_index + row_index * self._len

    def new_block(self, size):
        return self._db_allocate(self._len * size)

    def append(self, **data):
        return self._db_append(self._to_bytes(data))

    def get(self, block_index, row_index=0):
        index = self._get_index_from(block_index, row_index)
        data_bytes = self._read_at(index, 1)
        identifier = frombuffer(data_bytes, dtype="int8")[0]
        if identifier != self._identifier:
            raise KeyError
        data_bytes = self._read_at(index + self._prefix_size,
                                   self._len - self._prefix_size)
        return self._parse(data_bytes)

    def get_value(self, block_index, row_index, key):
        dt_size, align, dt = self._field[key]
        index = self._get_index_from(block_index, row_index) + align
        data_bytes = self._read_at(index, dt_size)
        res = frombuffer(data_bytes, dtype=dt)[0]
        return res

    def set(self, block_index, row_index, data):
        data_bytes = self._to_bytes(data)
        index = self._get_index_from(block_index, row_index)
        self._write_at(index, data_bytes)

    def set_value(self, block_index, row_index, key, value):
        _, align, dt = self._field[key]
        index = self._get_index_from(block_index, row_index) + align

        data = array(value, dtype=dt).tobytes()
        self._write_at(index, data)

    def __getitem__(self, args):
        if isinstance(args, tuple):
            arg_len = len(args)
            if arg_len == 2:
                return self.get(*args)
            elif arg_len == 3:
                return self.get_value(*args)
        return self._get_field_no_index(args)

    def __setitem__(self, args, value):
        if isinstance(args, tuple):
            arg_len = len(args)
            if arg_len == 2:
                return self.set(*args, value)
            elif arg_len == 3:
                return self.set_value(*args, value)
        return self._set_field_no_index(args, value)


    # def _build(self):
    #     field_to_dtype = {}
    #     self.size_align_dtype = {}
    #     self.string_fields = set()
    #     self.blob_fields = set()
    #     cum_pos = self.prefix_size
    #     for key, dt in self.dtypes:
    #         item_size = dt.itemsize
    #         field_to_dtype[key] = dt
    #         self.size_align_dtype[key] = (item_size, cum_pos, dt)

    #         if dt != blob_dt:  # item is regular field
    #             cum_pos += item_size
    #             if dt.type is str_:
    #                 self.string_fields.add(key)
    #         else:  # item is blob
    #             cum_pos += 4
    #             self.blob_fields.add(key)
    #     self.len = cum_pos
    #     self.field_to_dtype = field_to_dtype
    #     self.fields = list(self.field_to_dtype.keys())
    #     self.has_blob = len(self.blob_fields) != 0

    # def _finalize(self):
    #     self.size = self.to_unit(self.len)
    #     self.at = At(self)

    # def parse_blob(self, data):
    #     for field in self.blob_fields:
    #         if field not in data:
    #             continue
    #         data[field] = self.db_append_blob(data[field])

    # def _get_item_inplace(self, field):
    #     size, align, dt = self.size_align_dtype[field]

    #     data_bytes = self.read_at(
    #         self.offset + align, size)
    #     res = frombuffer(data_bytes, dtype=dt)[0]
    #     return res

    # def _get_row(self, index):
    #     # check row is indeed of good type
    #     byte_index = self.from_unit(index)
    #     data_bytes = self.read_at(byte_index, 1)
    #     identifier = frombuffer(data_bytes, dtype="int8")[0]
    #     if identifier != self.identifier:
    #         raise KeyError
    #     byte_index += self.prefix_size
    #     data_bytes = self.read_at(byte_index, self.len - self.prefix_size)
    #     return self._parse(data_bytes)

    # def _get_item(self, index, field):
    #     size, align, dt = self.size_align_dtype[field]

    #     byte_index = self.from_unit(index)
    #     data_bytes = self.read_at(
    #         byte_index + align, size)
    #     res = frombuffer(data_bytes, dtype=dt)[0]
    #     return res

    # def _get_items(self, start, stop):
    #     n_items = stop - start

    #     byte_index = self.from_unit(start)
    #     data_bytes = self.read_at(byte_index, self.len * n_items)
    #     res = frombuffer(data_bytes, dtype=[("prefix", int8)] + self.dtypes)
    #     data = [{key: item[key] for key in self.fields} for item in res]
    #     return data

    # def _set_item_inplace(self, field, value):
    #     _, align, dt = self.size_align_dtype[field]
    #     data = array(value, dtype=dt)
    #     self.write_at(self.offset + align, data)

    # def _set_row(self, index, data):
    #     data_bytes = self.to_bytes(**data)
    #     byte_index = self.from_unit(index)
    #     self.write_at(byte_index, data_bytes)

    # def _set_item(self, index, field, value):
    #     _, align, dt = self.size_align_dtype[field]
    #     data = array(value, dtype=dt)
    #     byte_index = self.from_unit(index)
    #     self.write_at(byte_index + align, data)

    # def _set_rows(self, start, rows):
    #     byte_index = self.from_unit(start)
    #     data_bytes = b''.join(self.to_bytes(**row) for row in rows)
    #     self.write_at(byte_index, data_bytes)

    # def __delitem__(self, start):
    #     byte_index = self.from_unit(start)
    #     data_bytes = self.read_at(byte_index, 1)
    #     identifier = frombuffer(data_bytes, dtype="int8")[0]
    #     assert identifier == self.identifier
    #     self.write_at(byte_index, int8(-identifier).tobytes())

    # def __getitem__(self, key):
    #     if isinstance(key, str):
    #         return self._get_item_inplace(key)
    #     elif isinstance(key, (int, integer)):
    #         return self._get_row(key)
    #     elif isinstance(key, tuple):
    #         index, field = key
    #         return self._get_item(index, field)
    #     elif isinstance(key, slice):
    #         start = key.start or 0
    #         stop = key.stop
    #         assert stop is not None
    #         return self._get_items(start, stop)

    # def __setitem__(self, key, value):
    #     if isinstance(key, str):
    #         return self._set_item_inplace(key, value)
    #     elif isinstance(key, (int, integer)):
    #         return self._set_row(key, value)
    #     elif isinstance(key, tuple):
    #         index, field = key
    #         return self._set_item(index, field, value)
    #     elif isinstance(key, slice):
    #         start = key.start
    #         assert start is not None
    #         return self._set_rows(start, value)

    # def __iter__(self):
    #     for data in self.db.iter_dataset(self):
    #         yield data

    # def slice(self, start, end):
    #     for data in self.db.iter_dataset(self, start, end):
    #         yield data


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
