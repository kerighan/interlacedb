import os
from pickle import HIGHEST_PROTOCOL, dumps, loads
from numpy import array, ceil, dtype, frombuffer, int8, uint32, where
from .dataset import Row, blob_dt



class InterlaceDB:
    _skip_through = 10000

    def __init__(
        self,
        filename,
        allocation_size=10000,
        blob_protocol="pickle",
        adapt_byte_size=False
    ):
        self.filename = filename
        self.allocation_size = allocation_size
        self.rows = {}
        self.dstructs = {}
        self._id2size = {}
        self._id2row = {}
        self._index = None
        self._adapt_byte_size = adapt_byte_size

        if not os.path.exists(filename) or self.file_size == 0:
            self.f = open(filename, "wb+")
        else:
            self.f = open(filename, "rb+")
            self.initialize()

        if blob_protocol == "pickle":
            self.encode = lambda x: dumps(x, protocol=HIGHEST_PROTOCOL)
            self.decode = loads
        elif blob_protocol == "ujson":
            import ujson
            self.encode = lambda x: bytes(ujson.dumps(x), "utf8")
            self.decode = lambda x: ujson.loads(str(x, "utf8"))
        elif blob_protocol == "orjson":
            import orjson
            self.encode = orjson.dumps
            self.decode = orjson.loads

    @property
    def index(self):
        if self._index is None:
            res = self.header["_index"]
            self._index = res
            return res
        return self._index

    @index.setter
    def index(self, value):
        value = int(value)
        self.header["_index"] = value
        self._index = value

    @property
    def file_size(self):
        return os.stat(self.filename).st_size

    @property
    def capacity(self):
        return (self.file_size - self.table_start) // self.unit_size

    @property
    def n_empty_slots(self):
        fs = os.stat(self.filename).st_size
        return int((fs - self.table_start) // self.unit_size - self.index)
    
    def initialize(self):
        table_start = 4

        # initialize header
        if not hasattr(self, "header"):
            self.create_header()
        self.f.seek(0)
        fields_len = frombuffer(self.f.read(4), dtype=uint32)[0]
        fields = loads(self.f.read(fields_len))
        self.build_header(fields)
        table_start += fields_len

        # initialize rows
        rows_len = frombuffer(self.f.read(4), dtype=uint32)[0]
        rows = loads(self.f.read(rows_len))
        for i, (row_name, dtypes) in enumerate(rows, 3):
            row = Row(i, self, row_name, dtypes)
            self.rows[row_name] = row
            self._id2size[i] = row.len
            self._id2row[i] = row
        table_start += rows_len + 4

        if self._adapt_byte_size or len(self.rows) == 1:
            self.unit_size = min(len(r) for r in self.rows.values())
        else:
            self.unit_size = 1

        self.pickle_size = table_start
        self.table_start = table_start + len(self.header)
        self.header.offset = self.pickle_size
        # finalize rows (add ``at method)
        for row in self.rows.values():
            row._finalize()

    def build_header(self, fields):
        if "_index" not in fields:
            dtypes = [("_index", dtype("uint64"))]
        else:
            dtypes = []
        dtypes += [(key, dtype(dt)) for key, dt in fields.items()]
        self.header = Row(1, self, "header", dtypes)

    def create_header(self, **fields):
        if self.file_size != 0:
            raise ValueError("Header already exists")
        
        # add datastructures header requests
        for dstruct in self.dstructs.values():
            new_fields = dstruct._get_header()
            for key, value in new_fields.items():
                assert key not in fields
                fields[key] = value

        # build header
        self.build_header(fields)
        fields_bytes = dumps(fields, protocol=HIGHEST_PROTOCOL)
        fields_len = array(len(fields_bytes), dtype=uint32).tobytes()

        self._extend_file(len(fields_len + fields_bytes))
        self.write_at(0, fields_len + fields_bytes)
        return self.header

    def create_dataset(self, row_name, **kwargs):
        assert row_name not in self.rows

        row_identifier = len(self.rows) + 2
        dtypes = []
        for key, dt in kwargs.items():
            if not isinstance(dt, str) or dt != "blob":
                dtypes.append((key, dtype(dt)))
            else:
                dtypes.append((key, blob_dt))

        row = Row(row_identifier, self, row_name, dtypes)
        self.rows[row_name] = row
        self._id2size[row_identifier] = row.len
        self._id2row[row_identifier] = row
        return row
    
    def create_datastructure(self, name, struct):
        assert name not in self.rows
        self.dstructs[name] = struct
        return struct

    def build(self):
        if not hasattr(self, "header"):
            self.create_header()

        row_data = []
        for row_name, row in self.rows.items():
            row_data.append((row_name, row.dtypes))
        # rows dumped into bytes
        row_bytes = dumps(row_data, protocol=HIGHEST_PROTOCOL)
        row_len = array(len(row_bytes), dtype=uint32).tobytes()
        # write rows on file
        offset = self.file_size
        self._extend_file(len(row_len + row_bytes))
        self.write_at(offset, row_len + row_bytes)
        self.pickle_size = self.file_size

        # initialize empty header
        self.header.offset = self.pickle_size
        header_bytes = self.header.to_bytes()
        self._extend_file(len(header_bytes))
        self.write_at(self.pickle_size, header_bytes)

        # offset table_start
        self.table_start = self.file_size
        self.unit_size = min(len(r) for r in self.rows.values())
        self.index = 0
        # finalize rows (add ``at method)
        for row in self.rows.values():
            row._finalize()
        # finalize 
        for dstruct in self.dstructs:
            dstruct._finalize()

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.build()

    def write_at(self, index, data):
        self.f.seek(index)
        self.f.write(data)

    def read_at(self, start, size):
        self.f.seek(start)
        return self.f.read(size)

    def put(self, index, data):
        byte_index = self.from_unit(index)
        self.write_at(byte_index, data)

    def append(self, data_bytes):
        data_size = self.to_unit(len(data_bytes))
        if self.n_empty_slots < data_size:
            alloc_size = int(self.allocation_size * self.unit_size)
            self._extend_file(max(alloc_size, data_size))

        index = self.index
        self.put(index, data_bytes)
        self.index += data_size
        return index

    def append_blob(self, blob):
        blob_bytes = self.encode(blob)
        blob_size = len(blob_bytes)

        data_bytes = b''.join((
            int8(1).tobytes(),  # 1 is blob identifier
            uint32(blob_size).tobytes(),
            blob_bytes))
        return self.append(data_bytes)

    def get_blob(self, index):
        byte_index = self.from_unit(index) + 1
        size = frombuffer(self.read_at(byte_index, 4), dtype=uint32)[0]
        blob_bytes = self.f.read(size)
        return self.decode(blob_bytes)

    def to_unit(self, bytes_size):
        return int(ceil(bytes_size / self.unit_size))

    def from_unit(self, index):
        return int(index * self.unit_size + self.table_start)

    def _extend_file(self, bytes_size):
        self.f.truncate(self.file_size + bytes_size)

    def allocate(self, size):
        # returns start and end indices of allocated data
        bytes_size = self.from_unit(size)
        self._extend_file(bytes_size)
        start = self.index
        self.index += size
        return start, self.index
    
    def read_slice(self, start, stop):
        start = self.from_unit(start)
        stop = self.from_unit(stop)
        return self.read_at(start, stop - start)

    def __getitem__(self, name):
        try:
            return self.rows[name]
        except KeyError:
            return self.dstructs[name]
    
    def __iter__(self):
        _skip_through = self._skip_through
        index = self.table_start
        limit = self.file_size - self.table_start

        while True:
            if index >= limit:
                return
            identifier = frombuffer(self.read_at(index, 1), dtype="int8")[0]
            if identifier == 0:
                arr = frombuffer(self.read_at(
                    index, _skip_through), dtype="int8")
                non_zeros = where(arr != 0)[0]
                if len(non_zeros) == 0:
                    index += _skip_through
                else:
                    index += non_zeros[0]
                continue
            elif identifier == 1:
                size = frombuffer(self.read_at(index + 1, 4), dtype=uint32)[0]
                blob_bytes = self.f.read(size)
                yield self.decode(blob_bytes)
                index += int(self.unit_size * self.to_unit(size))
                continue
            elif identifier < 0:  # element is deleted
                identifier = -identifier
                size = self._id2size[identifier]
                index += int(self.unit_size * self.to_unit(size))
                continue

            row = self._id2row[identifier]
            size = self._id2size[identifier]
            upper_size = int(self.unit_size * self.to_unit(size))

            data = row._parse(self.read_at(index + 1, size - 1))
            yield data

            index += upper_size
    
    def iter_dataset(self, row, start=None, end=None):
        _skip_through = self._skip_through
        if isinstance(row, Row):
            target = row.identifier
        else:
            target = row
            row = self._id2row[target]

        if start is None:
            index = self.table_start
        else:
            index = self.from_unit(start)
        if end is None:
            limit = self.file_size - self.table_start
        else:
            limit = min(self.file_size - self.table_start, self.from_unit(end))

        while True:
            if index >= limit:
                return
            identifier = frombuffer(self.read_at(index, 1), dtype="int8")[0]
            if identifier == 0:
                arr = frombuffer(self.read_at(index, _skip_through), dtype="int8")
                non_zeros = where(arr != 0)[0]
                if len(non_zeros) == 0:
                    index += _skip_through
                else:
                    index += non_zeros[0]
                continue
            elif identifier == 1:
                size = frombuffer(self.read_at(index + 1, 4), dtype=uint32)[0]
                index += int(self.unit_size * self.to_unit(size))
                continue
            elif identifier < 0:  # element is deleted
                identifier = -identifier
                size = self._id2size[identifier]
                index += int(self.unit_size * self.to_unit(size))
                continue

            size = self._id2size[identifier]
            upper_size = int(self.unit_size * self.to_unit(size))
            if identifier != target:
                index += upper_size
                continue
            
            data = row._parse(self.read_at(index + 1, size - 1))
            yield data

            index += upper_size
