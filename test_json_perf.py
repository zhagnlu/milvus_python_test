from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import time
import numpy


fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="json1", dtype=DataType.JSON, nullable=True),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]

def init_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  utility.drop_collection("hello_milvus")
  print("drop table done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  print("create table done")
  import random
  
  values = []
  array_len = 100
  index = 0
  while index < 100:
    for _id in range(index*100, (index+1)* 100):
      float_array = [float(_id) for _ in range(array_len)]
      varchar_array = [str(_id).zfill(100) for _ in range(array_len)]
      embedding = [float(_id) * random.random() for _ in range(128)]
      #values.append({"id": _id, "json1": {"float": float_array, "varchar": varchar_array}, "embeddings": embedding})
      values.append({"id": _id, "json1": "", "embeddings": embedding})
    #print(values)
    insert_result = hello_milvus.insert(values)  
    index +=1
    values = []
    print("insert index {}".format(index))
  # After final entity is inserted, it is best to call flush to have no growing segments left in memory
  #hello_milvus.flush()
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
  index_pq = {
              "index_type": "IVF_PQ",
              "metric_type": "L2",
            "params": { "m": 32, "nlist": 128},
   }
  #hello_milvus.create_index("embeddings", index)
  hello_milvus.create_index("embeddings", index)
  #hello_milvus.create_index("string1", index_name="string1_index")
  #hello_milvus.create_index("string2", index_name="string2_index")
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

def release_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.release()
  print("release done")
  
def load_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")

def search():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
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
  #result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["random"])
  #result = hello_milvus.query(expr="random in[1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0]", output_fields=["embeddings"])
#  while (True):
#    start = time.time()
#    result = hello_milvus.query(expr=" double1 > 0 and double1 < 3 ", output_fields=["double1"])
#    elapse_time = (time.time() - start) * 1000000
#    print(elapse_time)
  index = 0
  elapse_time = 0.0
  while (True):
    start = time.perf_counter()
    #result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit= 1000000, expr="double1 > 0", output_fields=["embeddings"])
    result = hello_milvus.search(vectors_to_search, "embeddings", search_params_hnsw, limit=3, expr=' 0< int1 < 3', output_fields=['embeddings'])
    print(len(result[0]))
    #print(result)
    #result = hello_milvus.query(expr="double1 < -300 and double1 >0 ", limit=1, output_fields=["int1"])
    #result = hello_milvus.query(expr=" double1 > -2 and double1 < -1 ", output_fields=["double1"])
    #for i in result:
        #print(i)
    #result = hello_milvus.query(expr="", output_fields=["count(*)"])
    #result = hello_milvus.query(expr=" double1 > -5 and double1 < 6 ", output_fields=["string1", "embeddings"])
    elapse_time += (time.perf_counter() - start) * 1000000
    index +=1
    #if index ==1 :
    #  exit(0)
    if (index % 10 == 0):
      print("avg:")
      print(elapse_time/10.0)
      elapse_time = 0
      insert_data(10)
    #print(elapse_time)
  #print(result)
  print("search done")

#expr = f"pk in [{ids[0]}, {ids[1]}]"
#hello_milvus.delete(expr)
#utility.drop_collection("hello_milvus")

def query():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")
  for _ in range(10000):
    res = hello_milvus.query(expr='json_contains_any(json1["float"], [100.0])')
    print(res)

if __name__ == "__main__":
  init_collection()  
  release_collection()
  load_collection()
  query()

