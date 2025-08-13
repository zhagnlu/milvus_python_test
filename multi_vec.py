import random
import time
import string

import pandas as pd
from sklearn import preprocessing
import datetime
import threading

from pymilvus import (
    connections, list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility, Partition, RRFRanker, WeightedRanker, AnnSearchRequest
)


def test_multi_vec(dim, nb):
    # for col in list_collections():
    #     oldColl = Collection(name=col)
    #     oldColl.drop()
    connections.connect("default", host="localhost", port="19530")
    print("connect done")
    utility.drop_collection("hello_milvus111")
    print("drop table done")

    default_fields = [
        FieldSchema(name="count", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="float_vector1", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="float_vector2", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="float_vector3", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="float_vector4", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    default_schema = CollectionSchema(fields=default_fields, description="test collection")

    collection = Collection(name="hello_milvus111", schema=default_schema, shards_num=1)

    vec_data = [[random.random() for _ in range(dim)] for _ in range(nb)]

    index = 0
    while index < 10:
        # insert data
        data = [
            [index * nb + i for i in range(nb)],
            vec_data,
            vec_data,
            vec_data,
            vec_data,
        ]
        res = collection.insert(data)
        print("insert done", index)
        index += 1
    collection.flush()

    metric_type = "L2"
    index_type = 'IVF_FLAT'
    nlist = 1024
    index_param = {
        "metric_type": metric_type,
        "index_type": index_type,
        "params": {"nlist": nlist}
    }

    # create vector index
    res = collection.create_index("float_vector1", index_param)
    print("create index done for float vector1")

    # create vector index
    res = collection.create_index("float_vector2", index_param)
    print("create index done for float vector2")

    # create vector index
    res = collection.create_index("float_vector3", index_param)
    print("create index done for float vector2")

    # create vector index
    res = collection.create_index("float_vector4", index_param)
    print("create index done for float vector2")
    collection.load()
    print("collection load done")

    i = 0
    while i < 1:
        search_param1 = {
            "data": [vec_data[i]],
            "anns_field": "float_vector1",
            "param": {"metric_type": "L2", "offset": 1},
            "limit": 1000,
            "expr": " 0<count< 10000",
        }
        req1 = AnnSearchRequest(**search_param1)
        search_param2 = {
            "data": [vec_data[100]],
            "anns_field": "float_vector2",
            "param": {"metric_type": "L2", "offset": 1},
            "limit": 10000,
            "expr": "30000 < count < 40000",
        }
        req2 = AnnSearchRequest(**search_param2)
        search_param3 = {
            "data": [vec_data[100]],
            "anns_field": "float_vector3",
            "param": {"metric_type": "L2", "offset": 1},
            "limit": 10000,
            "expr": "40000 <count < 50000",
        }
        req3 = AnnSearchRequest(**search_param3)
        search_param4 = {
            "data": [vec_data[100]],
            "anns_field": "float_vector4",
            "param": {"metric_type": "L2", "offset": 1},
            "limit": 10000,
            "expr": "50000 <count < 60000",
        }
        req4 = AnnSearchRequest(**search_param4)
        # res = collection.hybrid_search([req1, req2], WeightedRanker(0.9, 0.1), limit=2, output_fields=["float_vector1"])
        res = collection.hybrid_search([req1, req2, req3, req4], RRFRanker(), limit=10, output_fields=[], round_decimal=5)
        print(res)

    # print(res)

test_multi_vec(4000, 1000)

