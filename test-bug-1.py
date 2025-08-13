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
from concurrent.futures import ThreadPoolExecutor

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="string1", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="string2", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs", enable_dynamic_field=True)

def init_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")

  utility.drop_collection("hello_milvus")
  print("drop table done")

 
  hello_milvus = Collection("hello_milvus", schema)
  print("create table done")

  import random
  hello_milvus.create_index("string1", {"index_type" : "INVERTED"})
  hello_milvus.create_index("string2", {"index_type" : "INVERTED"})
  index = 0
  while(index < 50):
    start_pk = index * 3000  # 确保每次插入的 pk 都是唯一的
    
    # 使用列表模式批量插入数据
    entities = [
      [start_pk + i for i in range(3000)],  # field pk - 确保唯一性
      # [ "yyy" + str(i) if index < 50 else "xxx" + str(i) for i in range(3000)],  # field random
      # [ "aaa" + str(i) if index < 50 else "qqq" + str(i) for i in range(3000)],  # field random
      [ "abcdefg" if start_pk + i == 10000 or start_pk + i == 100000 else "xxx" + str(i) for i in range(3000)],  # field random
      [ "aaa" + str(i) if index < 50 else "qqq" + str(i) for i in range(3000)],  # field random
      [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
    ]
    insert_result = hello_milvus.insert(entities)
    index += 1
    print(index)
  # After final entity is inserted, it is best to call flush to have no growing segments left in memory
  hello_milvus.create_index("string1", {"index_type" : "INVERTED"})
  hello_milvus.create_index("string2", {"index_type" : "INVERTED"})
  hello_milvus.flush()
  print("insert and flush done")

  index = {
      "index_type": "IVF_FLAT",
      "metric_type": "L2",
      "params": {"nlist": 128},
  }
  index_hnsw = {
      "index_type": "HNSW",
      "metric_type": "IP",
            "params": {"M": 8, "efConstruction" : 256},
  }
  index_pq = {
              "index_type": "IVF_PQ",
              "metric_type": "L2",
            "params": { "m": 32, "nlist": 128},
   }
  #hello_milvus.create_index("embeddings", index)
  hello_milvus.create_index("embeddings", index_pq)
  hello_milvus.create_index("string1", {"index_type" : "INVERTED"})
  hello_milvus.create_index("string2", {"index_type" : "INVERTED"})
  #hello_milvus.create_index("double1", index_name="double1_index")
  #hello_milvus.create_index("double2", index_name="double2_index")
  print("create index done")

def load_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")

def perform_queries(num_queries  = 1):
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  hello_milvus = Collection("hello_milvus", schema)
  for _ in range(num_queries):
    # 由于没有动态字段，使用其他查询方式
    # result = hello_milvus.query(expr="string1 like '%xxx%' && string2 like '%qqq%' ", output_fields=["string1"])
    # print(len(result)) 
    # result = hello_milvus.query(expr="string2 like '%qqq%' && string1 like '%xxx%' ", output_fields=["string1"])
    # print(len(result))
    result = hello_milvus.query(expr="string2 == 'aaa1' or string2 == 'aaa2' or string1 == 'abcdefg'", output_fields=["string1", 'string2'])
    print(result) 
    
def perform_search(hello_milvus, num_queries):
  # connections.connect("default", host="localhost", port="19530")
  # print("connect done")
  # schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  # hello_milvus = Collection("hello_milvus", schema)
  import random
  for _ in range(num_queries):
    entities = [
      [i for i in range(3000)],  # field pk
      [float(random.randrange(-20, -10)) for _ in range(3000)],  # field random
      [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
    ]
    search_params = {
      "metric_type": "L2"
    }
    vectors_to_search = entities[-1][-2:]
    result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=10, expr="double2 > 0")
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
  vectors_to_search = entities[-1][-2:]
  search_params = {
      "metric_type": "L2",
      "params": {"nprobe": 10},
  }
  search_params_hnsw = {
      "metric_type": "IP",
      "params": {"ef": 50},
  }
  total_queries = 0
  start_time = time.time()
  with ThreadPoolExecutor(max_workers=num_threads) as executor:
    while time.time() - start_time < qps_test_duration:
        start = time.perf_counter()

        # Perform a batch of queries concurrently using threads
        futures = [executor.submit(perform_search, hello_milvus, queries_per_batch) for _ in range(num_threads)]
        # Wait for all threads to complete
        for future in futures:
            future.result()

        elapsed_time = time.perf_counter() - start
        total_queries += num_threads * queries_per_batch

        print(f"Batch QPS: { elapsed_time * 1000000 / (num_threads * queries_per_batch):.2f}")
  print(f"Total QPS: { qps_test_duration * 1000000 / total_queries:.2f}")

if __name__ == "__main__":
  init_collection()  # 取消注释以重新创建集合
  load_collection()
  perform_queries()


