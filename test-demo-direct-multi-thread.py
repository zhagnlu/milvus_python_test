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
import random
import threading
from concurrent.futures import ThreadPoolExecutor

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="int1", dtype=DataType.INT64),
    FieldSchema(name="int2", dtype=DataType.INT64),
    FieldSchema(name="int8_1", dtype=DataType.INT8),
    FieldSchema(name="int8_2", dtype=DataType.INT8),
    FieldSchema(name="float1", dtype=DataType.FLOAT),
    FieldSchema(name="float2", dtype=DataType.FLOAT),
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

  utility.drop_collection("hello_milvus")
  print("drop table done")

  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  print("create table done")

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
  hello_milvus.create_index("embeddings", index_hnsw)

  import random
  index = 0
  while(index < 40):
    entities = [
        [i for i in range(3000 *index, 3000 * (index + 1))],  # field pk
        [int(random.randrange(100, 300)) for _ in range(3000)],  # field int1
        [int(random.randrange(-30000, 30000)) for _ in range(3000)],  # field int2
        [numpy.int8(random.randrange(-120, 110)) for _ in range(3000)],  # field int8_1
        [numpy.int8(random.randrange(-80, 100)) for _ in range(3000)],  # field int8_2
        [float(random.randrange(-20, 100)) for _ in range(3000)],  # field double1
        [float(random.randrange(-20, 80)) for _ in range(3000)],  # field double2
        [float(random.randrange(-20, 100)) for _ in range(3000)],  # field double1
        [float(random.randrange(-20, 80)) for _ in range(3000)],  # field double2
              [ True if i %2 ==0 else False for i in range(3000)],  # field bool1 
              [ True if i %2 ==0 else False for i in range(3000)],  # field bool2
        [ "xxx" + str(i) for i in range(3000)],  # field random
        [ "xxx" + str(i) for i in range(3000)],  # field random
        [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
    ]
    if index == 0:
      print(entities[0])
    insert_result = hello_milvus.insert(entities)
    index += 1
  # After final entity is inserted, it is best to call flush to have no growing segments left in memory
  #hello_milvus.flush()
  print("insert and flush done")
  #hello_milvus.create_index("string1", index_name="string1_index")
  #hello_milvus.create_index("string2", index_name="string2_index")
  #hello_milvus.create_index("double1", index_name="double1_index")
  #hello_milvus.create_index("double2", index_name="double2_index")
  #print("create index done")

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
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")

def perform_search(hello_milvus, num_queries):
    while True:
        for _ in range(num_queries):
            import random
            #result = hello_milvus.query(expr="pk in [0, 1, 3,5, 6, 7, 8, 10, 11]",  output_fields=["int1"])
            vectors_to_search = [[random.random() for _ in range(128)] for _ in range(2)]
            search_params_hnsw = {
                  "metric_type": "L2",
                  "params": {"ef": 50},
              }
            #result = hello_milvus.search(vectors_to_search, "embeddings", search_params_hnsw, limit=3, expr='double1 > 0', output_fields=['embeddings'])
            hello_milvus.load()
            result = hello_milvus.search(vectors_to_search, "embeddings", search_params_hnsw, limit=3, expr='double1 > 0')
            print(len(result))

def perform_queries(hello_milvus, num_queries):
    while True:
        for _ in range(num_queries):
            hello_milvus.load()
            result = hello_milvus.query(expr="pk in [0, 1, 3,5, 6, 7, 8, 10, 11]")
            print(len(result))

def upsert(hello_milvus):
      index = 0
      while True:
          while index < 100:
              data = [
                  [i for i in range(3 * index, 3 * index + 3)],  # field pk
                  [int(random.randrange(100, 300)) for _ in range(3)],  # field int1
                  [int(random.randrange(-30, 30)) for _ in range(3)],  # field int2
                  [numpy.int8(random.randrange(-120, 110)) for _ in range(3)],  # field int8_1
                  [numpy.int8(random.randrange(-80, 100)) for _ in range(3)],  # field int8_2
                  [float(random.randrange(-20, 100)) for _ in range(3)],  # field double1
                  [float(random.randrange(-20, 80)) for _ in range(3)],  # field double2
                  [float(random.randrange(-20, 100)) for _ in range(3)],  # field double1
                  [float(random.randrange(-20, 80)) for _ in range(3)],  # field double2
                        [ True if i %2 ==0 else False for i in range(3)],  # field bool1 
                        [ True if i %2 ==0 else False for i in range(3)],  # field bool2
                  [ "xxx" + str(i) for i in range(3)],  # field random
                  [ "xxx" + str(i) for i in range(3)],  # field random
                  [[random.random() for _ in range(128)] for _ in range(3)],  # field embeddings
              ]
              hello_milvus.upsert(data) 
              index = index + 1 #result = hello_milvus.query(expr="pk in [0, 1, 3,5, 6, 7, 8, 10, 11]",  output_fields=["int1"])
          index=0

def search(qps_test_duration=40, queries_per_batch=50, num_threads=100):
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  vectors_to_search = [[random.random() for _ in range(128)] for _ in range(2)]
  search_params_hnsw = {
        "metric_type": "L2",
        "params": {"ef": 50},
    }
  result = hello_milvus.search(vectors_to_search, "embeddings", search_params_hnsw, limit=3, expr='double1 > 0') 
  print(len(result))
  result = hello_milvus.query(expr="pk in [0, 1, 3,5, 6, 7, 8, 10, 11]") 
  print(len(result))
  #while True:
  #  result = hello_milvus.query(expr="-1 < float1 < 100 ", limit=100, output_fields=["int1"])
  #  print(len(result))
  #print(result)
  while True:
      thread1 = threading.Thread(target=upsert, args=(hello_milvus,))
      thread1.start()
      thread2 = threading.Thread(target=perform_queries, args=(hello_milvus, 10))
      thread2.start()
      thread3 = threading.Thread(target=perform_search, args=(hello_milvus, 10))
      thread3.start()
      thread1.join()
      thread2.join()
      thread3.join()
  #start_time = time.time()
  #with ThreadPoolExecutor(max_workers=num_threads) as executor:
  #  while time.time() - start_time < qps_test_duration:
  #      start = time.perf_counter()

  #      # Perform a batch of queries concurrently using threads
  #      futures = [executor.submit(perform_queries, hello_milvus, queries_per_batch) for _ in range(num_threads)]
  #      # Wait for all threads to complete
  #      for future in futures:
  #          future.result()

  #      elapsed_time = time.perf_counter() - start
  #      total_queries += num_threads * queries_per_batch

  #      print(f"Batch QPS: { elapsed_time * 1000000 / (num_threads * queries_per_batch):.2f}")
  print(f"Total QPS: { qps_test_duration * 1000000 / total_queries:.2f}")
#
if __name__ == "__main__":
  init_collection()  
  load_collection()
  search(100, 2, 10)

