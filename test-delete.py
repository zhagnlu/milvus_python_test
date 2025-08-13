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
import threading


fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="int1", dtype=DataType.INT64),
    FieldSchema(name="str1", dtype=DataType.VARCHAR, max_length=2000),
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
 
  index = 0
  while(index < 4):
    entities = [
      [i + index * 30000 for i in range(30000)],  # field pk
      [int(random.randrange(100, 300)) for _ in range(30000)],  # field int1
      [ "xxx" + str(i) for i in range(30000)],  # field str1
      [[random.random() for _ in range(128)] for _ in range(30000)],  # field embeddings
     ]
    insert_result = hello_milvus.insert(entities)
    index += 1
  # After final entity is inserted, it is best to call flush to have no growing segments left in memory
  hello_milvus.flush()
  print("insert and flush done")

  index_hnsw = {
      "index_type": "HNSW",
      "metric_type": "L2",
            "params": {"M": 8, "efConstruction" : 256},
  }
  hello_milvus.create_index("embeddings", index_hnsw)
  print("create index done")

def insert_and_delete():
    schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
    hello_milvus = Collection("hello_small", schema)
    index = 0
    target = 1000
    while (index < target):
      import random
      entities = [
        [i + index * 30000 for i in range(30000)],  # field pk
        [int(random.randrange(100, 300)) for _ in range(30000)],  # field int1
        [ "xxx" + str(i) for i in range(30000)],  # field str1
        [[random.random() for _ in range(128)] for _ in range(30000)],  # field embeddings
      ]
      hello_milvus.insert(entities)
      print("insert done")
      sets = [index * 30000 + i for i in range(3000)]
      result_str = ", ".join(str(v) for v in sets)
      expr = f"pk in [{result_str}]"
      #print(f"expr:{expr}")
      print("query start")
      res = hello_milvus.query(expr=expr, output_fields=["count(*)"],  consistency_level="Strong")
      print(res)
      hello_milvus.delete(expr)
      print("delete done")
      res = hello_milvus.query(expr=expr, output_fields=["count(*)"], consistency_level="Strong",)
      print(res)
      print("query done")
      index += 1
      
def upsert_data(target_index):
  import random
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_small", schema)
  index = 0
  max_val = 0
  while(index < target_index):
    entities = [
      [random.randrange(0, 10000) for i in range(3000)],  # field pk
      #[i for i in range(3000)],  # field pk
      [int(random.randrange(-30000, 30000)) for _ in range(3000)],  # field int1
      [[random.random() for _ in range(128)] for _ in range(3000)],  # field embeddings
    ]
    insert_result = hello_milvus.upsert(entities)
    #print(hello_milvus.num_entities)
    for _ in range(1):
      result = hello_milvus.query(expr="", output_fields=["count(*)"]) 
      print(result[0])
      print(result[0]['count(*)'])
      time.sleep(1)
    if (max_val > result[0]['count(*)']):
        print("return")
        #break
    if (int(result[0]['count(*)']) > max_val):
        max_val = int(result[0]['count(*)'])
    index = index + 1

def multi_upsert_data(num_threads, target_index):
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=upsert_data, args=(target_index, ))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


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

if __name__ == "__main__":
  init_collection()  
  load_collection()
  insert_and_delete()
  release_collection()