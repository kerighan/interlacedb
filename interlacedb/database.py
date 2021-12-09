import os
from pickle import HIGHEST_PROTOCOL, dumps, loads

from numpy import array, ceil, dtype, frombuffer, int8, uint32, where

from .dataset import Dataset, blob_dt
from .exception import DatasetExistsError, HeaderExistsError

STEP_SIZE = 10000


class InterlaceDB:
    _skip_through = 10000

    def __init__(
        self,
        filename,
        blob_protocol="pickle",
        flag="w"
    ):
        """
        (str) filename: string name of the database file
        (int) step_size: number of bytes added when table is full
        (str) blob_protocol: protocol defining encoding and decoding functions
        """
        self.filename = filename
        self._get_encoder_and_decoder(blob_protocol)

        # references to header, datasets and datastructures
        self.header = None
        self.datasets = {}
        self.datastructures = {}

        # internal list of header fields
        self._header_fields = {}

        self._id2size = {}
        self._id2dataset = {}

        # index of read/write head
        self._index = None

        # if not os.path.exists(filename) or self.file_size == 0:
        #     self.f = open(filename, "wb+")
        # else:
        #     self.f = open(filename, "rb+")
        #     self.initialize()

        if flag == "n" and os.path.exists(filename):
            os.remove(filename)

        # open file
        self.open()

    def open(self):
        # create new file if needed, else open in rb+ mode
        if self._is_new_database():
            self.f = open(self.filename, "wb+")
        else:
            self.f = open(self.filename, "rb+")
            self._load()

    def _get_encoder_and_decoder(self, blob_protocol):
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

    # =========================================================================
    # properties
    # =========================================================================

    @property
    def file_size(self):
        return os.stat(self.filename).st_size

    @property
    def capacity(self):
        return self.file_size - self.table_start

    @property
    def index(self):
        if self._index is None:
            res = self.header._get_field_no_index("_index")
            self._index = res
            return res
        return self._index

    @index.setter
    def index(self, value):
        value = int(value)
        self.header._set_field_no_index("_index", value)
        self._index = value

    @property
    def _n_empty_slots(self):
        fs = os.stat(self.filename).st_size
        return fs - self.index

    # def initialize(self):
    #     table_start = 4

    #     # initialize header
    #     self.f.seek(0)
    #     fields_len = frombuffer(self.f.read(4), dtype=uint32)[0]
    #     fields = loads(self.f.read(fields_len))
    #     self.build_header(fields)
    #     table_start += fields_len

    #     # initialize rows
    #     rows_len = frombuffer(self.f.read(4), dtype=uint32)[0]
    #     rows = loads(self.f.read(rows_len))
    #     for i, (row_name, dtypes) in enumerate(rows, 3):
    #         row = Row(i, self, row_name, dtypes)
    #         self.rows[row_name] = row
    #         self._id2size[i] = row.len
    #         self._id2row[i] = row
    #     table_start += rows_len + 4

    #     if self._adapt_byte_size or len(self.rows) == 1:
    #         self.unit_size = min(len(r) for r in self.rows.values())
    #     else:
    #         self.unit_size = 1

    #     self.pickle_size = table_start
    #     self.table_start = table_start + len(self.header)
    #     self.header.offset = self.pickle_size
    #     # finalize rows (add ``at method)
    #     for row in self.rows.values():
    #         row._finalize()

    # =========================================================================
    # datasets and header management
    # =========================================================================

    def _dump(self):
        # build header
        self.header = Dataset(1, self, "header",
                              list(self._header_fields.items()))

        # remove database reference in datasets for pickle
        self.header._remove_database_reference()
        for name in self.datasets:
            self.datasets[name]._remove_database_reference()

        # dump header and datasets data
        data_bytes = dumps({"header": self.header, "datasets": self.datasets},
                           protocol=HIGHEST_PROTOCOL)
        data_len_bytes = array(len(data_bytes), dtype=uint32).tobytes()
        pickle_bytes = data_len_bytes + data_bytes
        pickle_bytes_len = len(pickle_bytes)

        # extend file and write on file
        self._extend_file(pickle_bytes_len)
        self._write_at(0, pickle_bytes)

        # bring back database reference in datasets
        self.header._add_database_reference(self)
        for name in self.datasets:
            self.datasets[name]._add_database_reference(self)

        # initialize heads
        self._extend_file(len(self.header))
        self.header._offset = pickle_bytes_len
        self.table_start = self.header._offset + len(self.header)
        self.index = self.file_size

    def _load(self):
        self.f.seek(0)
        data_len = frombuffer(self.f.read(4), dtype=uint32)[0]
        data = loads(self.f.read(data_len))
        self.header = data["header"]
        self.datasets = data["datasets"]

        # bring back database reference in datasets
        self.header._add_database_reference(self)
        for name in self.datasets:
            self.datasets[name]._add_database_reference(self)

        # initialize heads
        self.header._offset = data_len + 4
        self.table_start = self.header._offset + len(self.header)

    def create_dataset(self, name, **kwargs):
        if name in self.datasets:
            raise DatasetExistsError("A dataset named '{name}' already exists")

        identifier = len(self.datasets) + 3
        dtypes = []
        for key, dt in kwargs.items():
            if not isinstance(dt, str) or dt != "blob":
                dtypes.append((key, dtype(dt)))
            else:
                dtypes.append((key, blob_dt))

        dset = Dataset(identifier, self, name, dtypes)
        self.datasets[name] = dset
        self._id2size[identifier] = len(dset)
        self._id2dataset[identifier] = dset
        return dset

    def create_header(self, **fields):
        if self.file_size != 0:
            raise HeaderExistsError("Header already exists")

        if "_index" not in self._header_fields:
            self._header_fields["_index"] = dtype("uint64")

        for field, dt in fields.items():
            self._header_fields[field] = dtype(dt)

    # =========================================================================
    # overloading methods
    # =========================================================================

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._dump()
        self.close()
        self.open()

    # =========================================================================
    # data manipulation methods
    # =========================================================================

    def _allocate(self, bytes_size):
        # returns start and end indices of allocated data
        self._extend_file(bytes_size)
        start = self.index
        self.index += bytes_size
        return start

    def _append(self, data_bytes):
        data_size = len(data_bytes)
        # if there's no place left in the file, truncate
        if self._n_empty_slots < data_size:
            self._extend_file(max(STEP_SIZE, data_size))

        index = self.index
        self._write_at(index, data_bytes)
        self.index += data_size
        return index

    # =========================================================================
    # file IO management methods
    # =========================================================================

    def _is_new_database(self):
        return not os.path.exists(self.filename) or self.file_size == 0

    def _extend_file(self, bytes_size):
        self.f.truncate(self.file_size + bytes_size)

    def _write_at(self, index, data):
        self.f.seek(index)
        self.f.write(data)

    def _read_at(self, start, size):
        self.f.seek(start)
        return self.f.read(size)

    def close(self):
        self.f.close()

        # fields_bytes = dumps(fields, protocol=HIGHEST_PROTOCOL)
        # fields_len = array(len(fields_bytes), dtype=uint32).tobytes()

    #     # add datastructures header requests
    #     for dstruct in self.dstructs.values():
    #         new_fields = dstruct._get_header()
    #         for key, value in new_fields.items():
    #             assert key not in fields
    #             fields[key] = value

    #     # build header
    #     self.build_header(fields)

    #     self._extend_file(len(fields_len + fields_bytes))
    #     self.write_at(0, fields_len + fields_bytes)
    #     return self.header

    # def create_dataset(self, row_name, **kwargs):
    #     assert row_name not in self.rows

    #     row_identifier = len(self.rows) + 3
    #     dtypes = []
    #     for key, dt in kwargs.items():
    #         if not isinstance(dt, str) or dt != "blob":
    #             dtypes.append((key, dtype(dt)))
    #         else:
    #             dtypes.append((key, blob_dt))

    #     row = Row(row_identifier, self, row_name, dtypes)
    #     self.rows[row_name] = row
    #     self._id2size[row_identifier] = row.len
    #     self._id2row[row_identifier] = row
    #     return row

    # def create_datastructure(self, name, struct):
    #     assert name not in self.rows
    #     self.dstructs[name] = struct
    #     return struct

    # def build(self):
    #     if not hasattr(self, "header"):
    #         self.create_header()

    #     row_data = []
    #     for row_name, row in self.rows.items():
    #         row_data.append((row_name, row.dtypes))
    #     # rows dumped into bytes
    #     row_bytes = dumps(row_data, protocol=HIGHEST_PROTOCOL)
    #     row_len = array(len(row_bytes), dtype=uint32).tobytes()
    #     # write rows on file
    #     offset = self.file_size
    #     self._extend_file(len(row_len + row_bytes))
    #     self.write_at(offset, row_len + row_bytes)
    #     self.pickle_size = self.file_size

    #     # initialize empty header
    #     self.header.offset = self.pickle_size
    #     header_bytes = self.header.to_bytes()
    #     self._extend_file(len(header_bytes))
    #     self.write_at(self.pickle_size, header_bytes)

    #     # offset table_start
    #     self.table_start = self.file_size
    #     self.unit_size = min(len(r) for r in self.rows.values())
    #     self.index = 0
    #     # finalize rows (add ``at method)
    #     for row in self.rows.values():
    #         row._finalize()
    #     # finalize
    #     for dstruct in self.dstructs:
    #         dstruct._finalize()

    # def put(self, index, data):
    #     byte_index = self.from_unit(index)
    #     self.write_at(byte_index, data)

    # def append_blob(self, blob):
    #     blob_bytes = self.encode(blob)
    #     blob_size = len(blob_bytes)

    #     data_bytes = b''.join((
    #         int8(1).tobytes(),  # 1 is blob identifier
    #         uint32(blob_size).tobytes(),
    #         blob_bytes))
    #     return self.append(data_bytes)

    # def get_blob(self, index):
    #     byte_index = self.from_unit(index) + 1
    #     size = frombuffer(self.read_at(byte_index, 4), dtype=uint32)[0]
    #     blob_bytes = self.f.read(size)
    #     return self.decode(blob_bytes)

    # def to_unit(self, bytes_size):
    #     return int(ceil(bytes_size / self.unit_size))

    # def from_unit(self, index):
    #     return int(index * self.unit_size + self.table_start)

    # def read_slice(self, start, stop):
    #     start = self.from_unit(start)
    #     stop = self.from_unit(stop)
    #     return self.read_at(start, stop - start)

    # def __getitem__(self, name):
    #     try:
    #         return self.rows[name]
    #     except KeyError:
    #         return self.dstructs[name]

    # def __iter__(self):
    #     _skip_through = self._skip_through
    #     index = self.table_start
    #     limit = self.file_size - self.table_start

    #     while True:
    #         if index >= limit:
    #             return
    #         identifier = frombuffer(self.read_at(index, 1), dtype="int8")[0]
    #         if identifier == 0:
    #             arr = frombuffer(self.read_at(
    #                 index, _skip_through), dtype="int8")
    #             non_zeros = where(arr != 0)[0]
    #             if len(non_zeros) == 0:
    #                 index += _skip_through
    #             else:
    #                 index += non_zeros[0]
    #             continue
    #         elif identifier == 1:
    #             size = frombuffer(self.read_at(index + 1, 4), dtype=uint32)[0]
    #             blob_bytes = self.f.read(size)
    #             yield self.decode(blob_bytes)
    #             index += int(self.unit_size * self.to_unit(size))
    #             continue
    #         elif identifier < 0:  # element is deleted
    #             identifier = -identifier
    #             size = self._id2size[identifier]
    #             index += int(self.unit_size * self.to_unit(size))
    #             continue

    #         row = self._id2row[identifier]
    #         size = self._id2size[identifier]
    #         upper_size = int(self.unit_size * self.to_unit(size))

    #         data = row._parse(self.read_at(index + 1, size - 1))
    #         yield data

    #         index += upper_size

    # def iter_dataset(self, row, start=None, end=None):
    #     _skip_through = self._skip_through
    #     if isinstance(row, Row):
    #         target = row.identifier
    #     else:
    #         target = row
    #         row = self._id2row[target]

    #     if start is None:
    #         index = self.table_start
    #     else:
    #         index = self.from_unit(start)
    #     if end is None:
    #         limit = self.file_size - self.table_start
    #     else:
    #         limit = min(self.file_size - self.table_start, self.from_unit(end))

    #     while True:
    #         if index >= limit:
    #             return
    #         identifier = frombuffer(self.read_at(index, 1), dtype="int8")[0]
    #         if identifier == 0:
    #             arr = frombuffer(self.read_at(
    #                 index, _skip_through), dtype="int8")
    #             non_zeros = where(arr != 0)[0]
    #             if len(non_zeros) == 0:
    #                 index += _skip_through
    #             else:
    #                 index += non_zeros[0]
    #             continue
    #         elif identifier == 1:
    #             size = frombuffer(self.read_at(index + 1, 4), dtype=uint32)[0]
    #             index += int(self.unit_size * self.to_unit(size))
    #             continue
    #         elif identifier < 0:  # element is deleted
    #             identifier = -identifier
    #             size = self._id2size[identifier]
    #             index += int(self.unit_size * self.to_unit(size))
    #             continue

    #         size = self._id2size[identifier]
    #         upper_size = int(self.unit_size * self.to_unit(size))
    #         if identifier != target:
    #             index += upper_size
    #             continue

    #         data = row._parse(self.read_at(index + 1, size - 1))
    #         yield data

    #         index += upper_size
