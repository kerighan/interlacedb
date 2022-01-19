import os

from .database import InterlaceDB


class DataFrame:
    def __init__(self, filename, df=None, **kwargs):
        if df is not None:
            if os.path.exists(filename):
                os.remove(filename)
            with InterlaceDB(filename, flag="n", **kwargs) as db:
                self.data = db.create_dataset("data", value="blob")
                db.create_header(len="uint32", start="uint32")
            self.db = db
            self.header = db.header
            self.write_dataframe(df)
        else:
            db = InterlaceDB(filename, flag="r")
            self.data = db.datasets["data"]
            self.header = db.header
            self.start = self.header["start"]

    def write_dataframe(self, df):
        from tqdm import tqdm
        self.start = self.data.new_block(len(df))

        self.header["len"] = len(df)
        self.header["start"] = self.start
        i = 0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            self.data[self.start, i] = {"value": dict(row)}
            i += 1

    def get_row(self, index):
        return self.data[self.start, index]["value"]

    def get_rows(self, start, end):
        import pandas as pd
        if end < 0:
            end = len(self) + end
        end = min(end, len(self))
        return pd.DataFrame((self.get_row(i) for i in range(start, end)),
                            index=range(start, end))

    def __getitem__(self, i):
        import pandas as pd
        if isinstance(i, int):
            return self.get_row(i)
        elif isinstance(i, slice):
            start = i.start
            stop = i.stop
            if start is None and stop is None:
                return self.get_rows(0, self.__len__())
            elif start is None and stop is not None:
                return self.get_rows(0, stop)
            elif start is not None:
                return self.get_rows(start, stop)
        elif isinstance(i, list):
            return pd.DataFrame((self.get_row(d) for d in i), index=i)

    def __len__(self):
        return self.header["len"]

    def chunk(self, batch_size=100):
        for start in range(0, len(self), batch_size):
            end = start + batch_size
            yield self.get_rows(start, end)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_row(i)


class Graph:
    def __init__(self, filename, cache_len=1000000):
        from lru import LRU
        self.out_cache = LRU(cache_len)
        self.in_cache = LRU(cache_len)

        if os.path.exists(filename):
            db = InterlaceDB(filename)
            nodes = db.datastructures["nodes"]
            edges = db.datastructures["edges"]
            node = db.datasets["node"]
            edge = db.datasets["edge"]
        else:
            from .datastructure import LayerTable, MultiLayerTable
            with InterlaceDB(filename) as db:
                node = db.create_dataset(
                    "node",
                    key="U15",
                    _out_table="uint64",
                    _in_table="uint64")
                edge = db.create_dataset(
                    "edge", node="U15")
                nodes = LayerTable(node, key="key",
                                   n_bloom_filters=50,
                                   probe_factor=.1,
                                   growth_factor=2,
                                   cache_len=cache_len)
                edges = MultiLayerTable(edge, key="node",
                                        n_bloom_filters=50,
                                        probe_factor=.1,
                                        p_init=0,
                                        growth_factor=4,
                                        cache_len=cache_len)
                db.create_datastructure("nodes", nodes)
                db.create_datastructure("edges", edges)
                db.create_header(n_nodes="uint64", n_edges="uint64")
        self.nodes = nodes
        self.edges = edges
        self._node = node
        self._edge = edge
        self._hash = nodes._hash
        self.header = db.header
        self.db = db

    def add_node(self, u, commit=True):
        data = {"key": u}
        key_hash = self._hash(u)

        found = False
        try:
            p, position = self.nodes.find_lookup_position(u, key_hash)
            found = True
        except KeyError:
            p, position = self.nodes.find_insert_position(u, key_hash)
        table_id = self.nodes.tables_id[p - self.nodes.p_init]

        if found:
            return table_id, position

        if commit:
            self.db.begin_transaction()

        out_table = self.edges.new_table()
        in_table = self.edges.new_table()
        data["_out_table"] = out_table
        data["_in_table"] = in_table

        self._node.set(table_id, position, data)
        self.header["n_nodes"] += 1
        self.nodes._insert_in_bloom(p, u)
        self.nodes.cache[u] = p, position

        if commit:
            self.db.end_transaction()
        return table_id, position

    def get_node_position(self, u):
        key_hash = self._hash(u)
        p, position = self.nodes.find_lookup_position(u, key_hash)
        table_id = self.nodes.tables_id[p - self.nodes.p_init]
        return table_id, position

    def get_out_table(self, u, u_t, u_pos):
        u_out_table = self.out_cache.get(u)
        if u_out_table is None:
            u_out_table = self._node.get_value(u_t, u_pos, "_out_table")
            self.out_cache[u] = u_out_table
        return u_out_table

    def get_in_table(self, u, u_t, u_pos):
        u_in_table = self.in_cache.get(u)
        if u_in_table is None:
            u_in_table = self._node.get_value(u_t, u_pos, "_in_table")
            self.in_cache[u] = u_in_table
        return u_in_table

    def add_edge(self, u, v):
        self.db.begin_transaction()

        u_t, u_pos = self.add_node(u, False)
        v_t, v_pos = self.add_node(v, False)

        u_out_table = self.get_out_table(u, u_t, u_pos)
        v_in_table = self.get_in_table(v, v_t, v_pos)

        new_u_out = self.edges.insert(u_out_table, {"node": v})
        new_v_in = self.edges.insert(v_in_table, {"node": u})

        if new_u_out > u_out_table:
            self._node.set_value(u_t, u_pos, "_out_table", new_u_out)
            self.out_cache[u] = new_u_out
        if new_v_in > v_in_table:
            self._node.set_value(v_t, v_pos, "_in_table", new_v_in)
            self.in_cache[v] = new_v_in

        self.header["n_edges"] += 1

        self.db.end_transaction()

    def neighbors(self, u):
        u_t, u_pos = self.get_node_position(u)
        u_out_table = self._node.get_value(u_t, u_pos, "_out_table")
        return list(self.edges.iterate(u_out_table, "node"))

    def predecessors(self, u):
        u_t, u_pos = self.get_node_position(u)
        u_in_table = self._node.get_value(u_t, u_pos, "_in_table")
        return list(self.edges.iterate(u_in_table, "node"))
