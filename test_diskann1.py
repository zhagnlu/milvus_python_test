import random
import pandas as pd
from sklearn import preprocessing
import threading

from pymilvus import (
    connections, list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility, Partition
)

metric_type = 'L2'
index_type = 'DISKANN'
nb = 10000
dim = 256

def connect():
    connections.connect(
      host='127.0.0.1', 
      port=19530,
      user='root',
      password='1qaz@WSX',
      secure=False,
    )
    print("connect done")

def drop_collection(col):
    print(col)
    oldColl = Collection(name=col)
    oldColl.drop()

def create_collection():
    #drop_collection("hello_milvus7")
    print("drop done")
    default_fields = [
        FieldSchema(name="count", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="a", dtype=DataType.INT64),
        FieldSchema(name="b", dtype=DataType.INT64),
        FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    default_schema = CollectionSchema(fields=default_fields, description="test collection")

    collection = Collection(name="hello_milvus7", schema=default_schema, shards_num=1)
    partition = Partition(collection, "new_partition")

# insert data
    j = 0
    while j < 300 * nb:
        vec_data = [[random.random() for _ in range(dim)] for _ in range(nb)]
        data = [
            [j + i for i in range(nb)],
            [j + i for i in range(nb)],
            [j + i for i in range(nb)],
            vec_data,
        ]
        res = partition.insert(data)
        print("insert part done")
        j += nb

    print("insert all done")

    index_param = {
        "metric_type": metric_type,
        "index_type": index_type,
        "params": {}
    }
    # create vector index
    res = collection.create_index("float_vector", index_param)
    print("create index done")
    return collection


def load_collection():
    default_fields = [
        FieldSchema(name="count", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="a", dtype=DataType.INT64),
        FieldSchema(name="b", dtype=DataType.INT64),
        FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    default_schema = CollectionSchema(fields=default_fields, description="test collection")

    collection = Collection(name="hello_milvus7", schema=default_schema, shards_num=1)
    print("start load collection")
    res = collection.load()
    print("load collection done")
    return collection

def worker(collection, expr_str):
    result = collection.query(expr=expr_str, output_fields=["float_vector"])

def search(collection):
    nq = 10
    search_params = {"metric_type": metric_type, "params": {"search_list": 150}}
    vec_data = [[random.random() for _ in range(dim)] for _ in range(nb)]
    results = collection.search(
        vec_data[:nq],
        anns_field="float_vector",
        param=search_params,
        limit=150,
      #  consistency_level="Strong",
        output_fields=["float_vector"]
    )
    #pks = ",".join( str(i) for i in range(50))
    #print(pks)
    #expr_str = "count in [" + pks + "]"
    #print(expr_str)
    #threads = []
    #for i in range(10):
    #    print(i)
    #    thread = threading.Thread(target=worker, args=(collection, expr_str))
    #    thread.start()
    #    threads.append(thread)
    #    #result = collection.query(expr=expr_str, output_fields=["float_vector"])
    #for i in range(10):
    #    threads[i].join()

    ##print(result)
    #print(len(results[0].ids))

if __name__ == "__main__":
    connect()
    #collection = create_collection()
    collection = load_collection()
    search(collection)
