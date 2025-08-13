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
    # FieldSchema(name="int2", dtype=DataType.INT64),
    # FieldSchema(name="int8_1", dtype=DataType.INT8),
    # FieldSchema(name="int8_2", dtype=DataType.INT8),
    # FieldSchema(name="double1", dtype=DataType.DOUBLE),
    # FieldSchema(name="double2", dtype=DataType.DOUBLE),
    # FieldSchema(name="bool1", dtype=DataType.BOOL),
    # FieldSchema(name="bool2", dtype=DataType.BOOL),
    FieldSchema(name="string1", dtype=DataType.VARCHAR, max_length=20000),
    #FieldSchema(name="string2", dtype=DataType.VARCHAR, max_length=20000),
    # FieldSchema(name="array1", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=5),
    # FieldSchema(name="array2", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=1000, max_capacity=5),
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
    # [int(random.randrange(-10, 900)) for _ in range(3000)],  # field int2
    # [numpy.int8(random.randrange(-120, 110)) for _ in range(3000)],  # field int8_1
    # [numpy.int8(random.randrange(-80, 100)) for _ in range(3000)],  # field int8_2
    # [float(random.randrange(-20, 100)) for _ in range(3000)],  # field double1
    # [float(random.randrange(-20, 80)) for _ in range(3000)],  # field double2
    #       [ True if i %2 ==0 else False for i in range(3000)],  # field bool1 
    #       [ True if i %2 ==0 else False for i in range(3000)],  # field bool2
    [ "xxx" + str(random.randint(0, 10000) % 1000) for i in range(3000)],  # field string1
    # [ "xxx" + str(random.randint(0, 10000) % 30) for i in range(3000)],  # field string2
    # [[int(random.uniform(0, 10000)) for _ in range(5) ] for _ in range(3000)], #array 1
    # [[str(random.uniform(0, 10000)) for _ in range(5) ] for _ in range(3000)], #array 2
    # [[float(random.randrange(-20, 100)) for _ in range(5) ] for _ in range(3000)], #array 3
    # [[True if random.randint(0, 10000) % 2 == 0  else False for _ in range(5) ] for _ in range(3000)], #array 4
    [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
]

index_param = { "index_type" : "BITMAP"}
#index_param = { "index_type" : "INVERTED"}
hello_milvus.create_index("int1", index_name='xxx', index_params=index_param)
# hello_milvus.create_index("int8_1", index_param)
# hello_milvus.create_index("bool1", index_param)
hello_milvus.create_index("string1", index_name='xx', index_params=index_param)
#hello_milvus.create_index("string1", index_name='xx', index_params=index_param)
print("create index done")
#hello_milvus.create_index("array1", index_param)
# hello_milvus.create_index("array2", index_param)
# hello_milvus.create_index("array4", index_param)
#hello_milvus.alter_index('xxx', {'mmap.enabled': True, 'indexoffsetcache.enabled': True})

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

#hello_milvus.create_index("embeddings", index)
hello_milvus.create_index("embeddings", index_hnsw)

index = 0
while(index < 50):
  insert_result = hello_milvus.insert(entities)
  index += 1
# After final entity is inserted, it is best to call flush to have no growing segments left in memory
hello_milvus.flush()
print("insert and flush done")

hello_milvus.alter_index('xxx', {'indexoffsetcache.enabled': True})
hello_milvus.alter_index('xx', {'indexoffsetcache.enabled': True})
hello_milvus.alter_index('xxx', {'mmap.enabled': True})
hello_milvus.alter_index('xx', {'mmap.enabled': True})
print("alter index done")
time.sleep(20)

#hello_milvus.alter_index('xx', {'mmap.enabled': True, 'indexoffsetcache.enabled': True})
hello_milvus.load()
print("load done")

index = 0
start_time = time.perf_counter()
while index < 1:
  #result = hello_milvus.search(vectors_to_search, "embeddingss", search_params, limit=100, expr='double1>0') 
  result = hello_milvus.query(expr="string1 like '%xxx1%'", output_fields=["pk", "string1"])
  print(len(result))
  result = hello_milvus.query(expr="int1 > 2000", output_fields=["pk", "int1"])
  print(len(result))
  #print(result)
  index +=1
  time.sleep(1)
elapsed_time = time.perf_counter()
print(f"Search time: {elapsed_time - start_time} seconds")

hello_milvus.release()
#index_param = { "index_type" : "BITMAP"}
hello_milvus.drop_index(index_name="xx")
hello_milvus.drop_index(index_name="xxx")
hello_milvus.load()
index =0
start_time = time.perf_counter()
while index < 1:
  #result = hello_milvus.search(vectors_to_search, "embeddingss", search_params, limit=100, expr='double1>0') 
  result = hello_milvus.query(expr="string1 like \"%xxx1%\"", output_fields=["pk", "string1"])
  print(len(result))
  result = hello_milvus.query(expr="int1 > 2000", output_fields=["pk", "int1"])
  print(len(result))
  #print(result)
  index +=1
  time.sleep(1)
elapsed_time = time.perf_counter()
print(f"Search time: {elapsed_time - start_time} seconds")

#index = hello_milvus.create_index("array1", index_name='xx',  index_params=index_param)
#print(index)
#i =  0
#while i < 10:
#    print(utility.index_building_progress("hello_small", 'xx'))
#    time.sleep(1)
#    i += 1
#hello_milvus.load()
#while True:
#    result = hello_milvus.query(expr="json_contains(array1, 100)", output_fields=["pk", "int2", "array1"])
#    print(result)
#    time.sleep(1)

#entities = [
#    [i for i in range(3000)],  # field pk
#    [float(random.randrange(-20, -10)) for _ in range(3000)],  # field random
#    [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
#]
#vectors_to_search = entities[-1][-2:]
#search_params = {
#    "metric_type": "L2",
#    "params": {"nprobe": 10},
#}
#search_params_hnsw = {
#    "metric_type": "L2",
#    "params": {"ef": 50},
#}
#
#index = 0
#while index < 1:
#    result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=100,
#                                 expr='array1[0] >= 5 ', output_fields=["pk", "int2", "array1"]) 
#    print (result)
#
