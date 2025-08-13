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
    FieldSchema(name="int2", dtype=DataType.INT64),
    FieldSchema(name="int8_1", dtype=DataType.INT8),
    FieldSchema(name="int8_2", dtype=DataType.INT8),
    FieldSchema(name="double1", dtype=DataType.DOUBLE),
    FieldSchema(name="double2", dtype=DataType.DOUBLE),
    FieldSchema(name="bool1", dtype=DataType.BOOL),
    FieldSchema(name="bool2", dtype=DataType.BOOL),
    FieldSchema(name="string1", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="string2", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]

def init_collection():
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
      [int(random.randrange(100, 300)) for _ in range(3000)],  # field int1
      [int(random.randrange(-10, 900)) for _ in range(3000)],  # field int2
      [numpy.int8(random.randrange(-120, 110)) for _ in range(3000)],  # field int8_1
      [numpy.int8(random.randrange(-80, 100)) for _ in range(3000)],  # field int8_2
      [float(random.randrange(-20, 100)) for _ in range(3000)],  # field double1
      [float(random.randrange(-20, 80)) for _ in range(3000)],  # field double2
            [ True if i %2 ==0 else False for i in range(3000)],  # field bool1 
            [ True if i %2 ==0 else False for i in range(3000)],  # field bool2
      [ "xxx" + str(i) for i in range(3000)],  # field random
      [ "xxx" + str(i) for i in range(3000)],  # field random
      [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
  ]
  
  #index_param = { "index_type" : "AUTOINDEX", "bitmap_cardinality_limit" : 1000}
  index_param = { "index_type" : ""}
  #index_param = { "index_type" : "AUTOINDEX"}
  hello_milvus.create_index("int2", index_param)
  hello_milvus.create_index("string1", index_param)

  index = 0
  while(index < 40):
    insert_result = hello_milvus.insert(entities)
    index += 1
  # After final entity is inserted, it is best to call flush to have no growing segments left in memory
  hello_milvus.flush()
  print("insert and flush done")

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
  #hello_milvus.create_index("int2", index_param)
  #hello_milvus.create_index("pk")
  #hello_milvus.create_index("string1", index_name="string1_index")
  #hello_milvus.create_index("string2", index_name="string2_index")
  #hello_milvus.create_index("double1", index_name="double1_index")
  #hello_milvus.create_index("double2", index_name="double2_index")
  print("create index done")

def insert_data(target_index):
  import random
  entities = [
      [i for i in range(3000)],  # field pk
      [int(random.randrange(-30000, 30000)) for _ in range(3000)],  # field int1
      [int(random.randrange(-30000, 30000)) for _ in range(3000)],  # field int2
      [numpy.int8(random.randrange(-127, 127)) for _ in range(3000)],  # field int8_1
      [numpy.int8(random.randrange(-127, 127)) for _ in range(3000)],  # field int8_2
      [float(random.randrange(-20, 10)) for _ in range(3000)],  # field double1
      [float(random.randrange(-20, 10)) for _ in range(3000)],  # field double2
            [ True if i %2 ==0 else False for i in range(3000)],  # field bool1 
            [ True if i %2 ==0 else False for i in range(3000)],  # field bool2
      [ "xxx" + str(i) for i in range(3000)],  # field random
      [ "xxx" + str(i) for i in range(3000)],  # field random
      [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
  ]
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  index = 0
  while(index < target_index):
    insert_result = hello_milvus.insert(entities)
    index += 1

def load_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_small", schema)
  hello_milvus.load()
  print("load done")

def release_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_small", schema)
  hello_milvus.release()
  print("release done")
  #index_param = { "index_type" : "BITMAP"}
  #hello_milvus.create_index("int2", index_param)

def search():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_small", schema)
  import random
  entities = [
      [i for i in range(3000)],  # field pk
      [float(random.randrange(-20, -10)) for _ in range(3000)],  # field random
      [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
  ]
  vectors_to_search = entities[-1][-2:]
  search_params = {
      "metric_type": "L2",
      "params": {"nprobe": 10},
  }
  search_params_hnsw = {
      "metric_type": "L2",
      "params": {"ef": 50},
  }

  index = 0
  while index < 1:
    #result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=100, expr='double1>0') 
    result = hello_milvus.query(expr="int2 in[1, 10000, 300000, 500000]", output_fields=["pk", "int2"])
    print(result)
    index +=1
#  while (True):
#    start = time.time()
#    result = hello_milvus.query(expr=" double1 > 0 and double1 < 3 ", output_fields=["double1"])
#    elapse_time = (time.time() - start) * 1000000
#    print(elapse_time)


#expr = f"pk in [{ids[0]}, {ids[1]}]"
#hello_milvus.delete(expr)
#utility.drop_collection("hello_milvus")

if __name__ == "__main__":
  init_collection()  
  load_collection()
  search()
  #release_collection()
  #load_collection()

