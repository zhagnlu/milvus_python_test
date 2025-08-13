from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import time
import string
import threading
import numpy
import random
from concurrent.futures import ThreadPoolExecutor

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=2000, is_primary=True, auto_id=False),
    FieldSchema(name="int1", dtype=DataType.INT64),
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

  index = 0
  while(index < 100):
    entities = [
      [''.join(random.choices(string.ascii_letters + string.digits, k=40)) for i in range(10000)],  # field pk
      [int(random.randrange(100, 300)) for _ in range(10000)],  # field int1
      [[random.random() for _ in range(128)] for _ in range(10000)],  # field embeddings
  ]
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
  index_pq = {
              "index_type": "IVF_PQ",
              "metric_type": "L2",
            "params": { "m": 32, "nlist": 128},
   }
  #hello_milvus.create_index("embeddings", index)
  hello_milvus.create_index("embeddings", index)
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
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")

def perform_queries(hello_milvus, num_queries):
    for _ in range(num_queries):
        result = hello_milvus.query(expr="pk not in [10, 11]", limit=100, output_fields=["int1"])

def upsert_data(start, target_index):

  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  index = 0 
  while(index < target_index):
    entities = [
      [start + i + index * 1000 for i in range(1000)],  # field pk
      [int(random.randrange(100, 300)) for _ in range(1000)],  # field int1
      [[random.random() for _ in range(128)] for _ in range(1000)],  # field embeddings
    ]
    print(entities[0])
    insert_result = hello_milvus.upsert(entities)
    index += 1

def run_upsert():
  while(True):
    upsert_data(100000, 100)
    time.sleep(1)

def query():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  #result = hello_milvus.query(expr=" 0<= pk < 100000", output_fields=["count(*)"])
  random_pk = ''.join(random.choices(string.ascii_letters + string.digits, k=40))
  result = hello_milvus.query(expr=f" pk not in ['{random_pk}']", output_fields=["count(*)"])
  print(result)

def search(qps_test_duration=40, queries_per_batch=50, num_threads=100):
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
  print(entities[0])
  vectors_to_search = entities[-1][-2:]
  search_params = {
      "metric_type": "L2",
      "params": {"nprobe": 10},
  }
  search_params_hnsw = {
      "metric_type": "L2",
      "params": {"ef": 50},
  }
  total_queries = 0
  start_time = time.time()
  with ThreadPoolExecutor(max_workers=num_threads) as executor:
    while time.time() - start_time < qps_test_duration:
        start = time.perf_counter()

        # Perform a batch of queries concurrently using threads
        futures = [executor.submit(perform_queries, hello_milvus, queries_per_batch) for _ in range(num_threads)]
        # Wait for all threads to complete
        for future in futures:
            future.result()

        elapsed_time = time.perf_counter() - start
        total_queries += num_threads * queries_per_batch

        print(f"Batch QPS: { elapsed_time * 1000000 / (num_threads * queries_per_batch):.2f}")
  print(f"Total QPS: { qps_test_duration * 1000000 / total_queries:.2f}")

if __name__ == "__main__":
  #init_collection()  
  load_collection()
  #t = threading.Thread(target=run_upsert, daemon=True)
  #t.start()
  #upsert_data(100000, 100)
  while(True):
    query()
    time.sleep(1)


