import os

from convectors.layers import Tokenize
from interlacedb import InterlaceDB
from interlacedb.datastructure import LayerTable, MultiLayerTable
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

newsgroups_train = fetch_20newsgroups(subset="all")
data = newsgroups_train.data
nlp = Tokenize(stopwords=["fr", "en"], verbose=False)

if not os.path.exists("engine.db"):
    with InterlaceDB("engine.db", flag="n") as db:
        word = db.create_dataset("word", key="U15", table="uint64")
        words = LayerTable(
            word, "key",
            p_init=14, probe_factor=.1,
            cache_len=1000000, n_bloom_filters=25)
        db.create_datastructure("words", words)

        entry = db.create_dataset("entry", index="uint32")
        entries = MultiLayerTable(
            entry, key="index", p_init=2,
            probe_factor=.1,
            cache_len=1000000, n_bloom_filters=25)
        db.create_datastructure("entries", entries)

    for i, doc in tqdm(enumerate(data), total=len(data)):
        tokens = set(nlp(doc))
        db.begin_transaction()
        for token in tokens:
            key_hash = words._hash(token)
            block_id, position, new = words.find_insert_or_lookup_index(
                token, key_hash)
            if not new:
                table_id = word[block_id, position, "table"]
                new_table_id = entries.insert(table_id, {"index": i})
                if new_table_id > table_id:
                    word[block_id, position, "table"] = new_table_id
            else:
                table_id = entries.insert(entries.new_table(), {"index": i})
                words[token] = {"table": table_id}

        db.end_transaction()
else:
    from random import shuffle
    db = InterlaceDB("engine.db", flag="r")
    words = db.datastructures["words"]
    entries = db.datastructures["entries"]

    w = list(words)
    print(len(w))
    w = [i["key"] for i in w]
    shuffle(w)

    for it in tqdm(w):
        if len(it) >= 15:
            continue
        table_id = words[it]["table"]
        doc = set(entries.iterate(table_id, field="index"))

    # while True:
    #     try:
    #         query = input("query: ")
    #         query = query.split()
    #         documents = set(entries.iterate(
    #             words[query[0]]["table"], field="index"))
    #         for j in range(1, len(query)):
    #             documents = documents.intersection(
    #                 set(entries.iterate(
    #                     words[query[j]]["table"], field="index")))

    #         documents = list(documents)
    #         for j in range(min(3, len(documents))):
    #             print(data[documents[j]])
    #             print()
    #         print(len(documents))
    #     except KeyError:
    #         words._load_tables_id()
    #         print(words.tables_id)
    #         print("Not found")
