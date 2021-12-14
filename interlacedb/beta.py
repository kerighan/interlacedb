import os

import pandas as pd
from numpy.core.shape_base import block
from numpy.lib.arraysetops import isin

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
            db = InterlaceDB(filename)
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
        if end < 0:
            end = len(self) + end
        end = min(end, len(self))
        return pd.DataFrame((self.get_row(i) for i in range(start, end)),
                            index=range(start, end))

    def __getitem__(self, i):
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
