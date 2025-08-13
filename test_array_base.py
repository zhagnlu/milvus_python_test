from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from pymilvus import MilvusClient, DataType


import time
import numpy

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
     FieldSchema(name="int1", dtype=DataType.INT64),
     FieldSchema(name="array1", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=5, nullable=True),
     FieldSchema(name="array2", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=1000, max_capacity=5),
    # FieldSchema(name="array3", dtype=DataType.ARRAY, element_type=DataType.DOUBLE, max_capacity=5),
    # FieldSchema(name="array4", dtype=DataType.ARRAY, element_type=DataType.BOOL, max_capacity=5),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]

connections.connect("default", host="localhost", port="19530")
print("connect done")
utility.drop_collection("hello_small")
print("drop table done")
schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
hello_milvus = Collection("hello_small", schema)
print("create table done")

import random
entities = [
    [i for i in range(3000)],  # field pk
     [int(random.randrange(100, 3000)) for _ in range(3000)],  # field int1
     [ [] if random.random() < 0.1 else [int(random.uniform(0, 10000)) for _ in range(5)]for _ in range(3000)], #array 1
     [[str(random.uniform(0, 10000)) for _ in range(5) ] for _ in range(3000)], #array 2
    # [[float(random.randrange(-20, 100)) for _ in range(5) ] for _ in range(3000)], #array 3
    # [[True if random.randint(0, 10000) % 2 == 0  else False for _ in range(5) ] for _ in range(3000)], #array 4
    [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
]

index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
index_hnsw = {
    "index_type": "HNSW",
    "metric_type": "L2",
          "params": {"M": 8, "efConstruction" : 256},
}
index_disk = {
    "index_type": "DISKANN",
    "metric_type": "L2",
          "params": {"M": 8, "efConstruction" : 256},
}
index_pq = {
            "index_type": "IVF_PQ",
            "metric_type": "L2",
          "params": { "m": 32, "nlist": 128},
 }

hello_milvus.create_index("embeddings", index_hnsw)

hello_milvus.load()
print("load done")

result = hello_milvus.query(expr="int1 > 2000", output_fields=["pk", "int1"])
print(result)
index = 0
while(index < 50):
  insert_result = hello_milvus.insert(entities)
  index += 1
# After final entity is inserted, it is best to call flush to have no growing segments left in memory
#hello_milvus.flush()
print("insert and flush done")

hello_milvus.load()

print("load done")