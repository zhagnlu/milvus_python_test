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
import string
import threading
import json
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

def query_task(i, coll, latency_list):
    values = [(i + j * 13) % 10000 for j in range(1000)]  # 用乘法散开一点
    expr = f"int2 in {values}"
    try:
        start = time.perf_counter()
        result = coll.query(expr=expr, output_fields=["pk"])
        duration = time.perf_counter() - start
        latency_list.append(duration)
        print(f"[Thread-{i}] Count: {len(result)}, Latency: {duration:.3f}s")
    except Exception as e:
        print(f"[Thread-{i}] Query failed: {e}")

def concurrent_query(coll, thread_num=10):
    latency_list = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(query_task, i, coll, latency_list) for i in range(thread_num)]
        for future in as_completed(futures):
            future.result()

    total_time = time.perf_counter() - start_time
    total_queries = len(latency_list)
    avg_latency = sum(latency_list) / total_queries if total_queries > 0 else 0
    qps = total_queries / total_time if total_time > 0 else 0

    print("\n=== Stats ===")
    print(f"Total Queries : {total_queries}")
    print(f"Total Time    : {total_time:.3f}s")
    print(f"Avg Latency   : {avg_latency:.3f}s")
    print(f"QPS           : {qps:.2f}")
            
def generate_random_string():
    length = random.randint(1, 1000)
    characters = string.ascii_letters
    random_string = ''.join(random.choice(characters) for _ in range(100))
    return random_string

# Generate a pool of 1000 random strings with length 10
STRING_POOL_FILE = "string_pool.json"

def load_or_generate_string_pool():
    """Load string pool from file if it exists, otherwise generate and save it"""
    if os.path.exists(STRING_POOL_FILE):
        try:
            with open(STRING_POOL_FILE, 'r') as f:
                string_pool = json.load(f)
            print(f"Loaded string pool from {STRING_POOL_FILE}")
            return string_pool
        except Exception as e:
            print(f"Failed to load string pool from file: {e}")
    
    # Generate new string pool
    string_pool = [''.join(random.choices(string.ascii_letters, k=10)) for _ in range(10000)]
    
    # Save to file
    try:
        with open(STRING_POOL_FILE, 'w') as f:
            json.dump(string_pool, f)
        print(f"Generated and saved string pool to {STRING_POOL_FILE}")
    except Exception as e:
        print(f"Failed to save string pool to file: {e}")
    
    return string_pool


string_pool = load_or_generate_string_pool()
#string_pool = [''.join(random.choices(string.ascii_letters, k=10)) for _ in range(10)]

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True,  auto_id=False),
    # FieldSchema(name="int1", dtype=DataType.INT64,is_partition_key=True),
    FieldSchema(name="int2", dtype=DataType.INT64),
    FieldSchema(name="string1", dtype=DataType.VARCHAR, max_length=200, nullable=True),
    # FieldSchema(name="json", dtype=DataType.JSON, max_length=20000),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
]


def random_value():
    """随机生成不同类型的值"""
    value_types = [
        lambda: random.randint(1, 1000),  # 随机整数
        lambda: round(random.uniform(1.0, 1000.0), 2),  # 随机浮点数
        lambda: ''.join(random.choices(string.ascii_letters, k=10)),  # 随机字符串
        lambda: [random.randint(1, 100) for _ in range(random.randint(2, 5))],  # 整数数组
        #lambda: [round(random.uniform(1.0, 100.0), 2) for _ in range(random.randint(2, 5))],  # 浮点数组
        #lambda: {f"subkey_{i}": random.randint(1, 100) for i in range(random.randint(1, 3))},  # 随机子对象
    ]
    return random.choice(value_types)()



def init_collection():
  connections.connect("default", host="localhost", port="19530")

  print("connect done")

  utility.drop_collection("hello_milvus")
  print("drop table done")

  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema, shards_num=1)
  print("create table done")

  #hello_milvus.create_index("string1", index_params={"index_type": "INVERTED"})
  print("create index done")

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
  while(index < 100):
    entities = [
        [i for i in range(10000 *index, 10000 * (index + 1))],  # field pk
        # [int(random.randrange(0, 1023)) for _ in range(100)],  # field int1
        [i for i in range(10000 *index, 10000 * (index + 1))],  # field int2
        [ "xxx" for i in range(10000)],  # field random
        # [ get_json(100) for i in range(100)],
        [[random.random() for _ in range(128)] for _ in range(10000)],  # field embeddings
    ]
    #if index == 0:
    #  print(entities)
    insert_result = hello_milvus.insert(entities)
    index += 1
    print(f"insert{index}")
  hello_milvus.flush()
  print("flush done")
  #hello_milvus.create_index("string1", index_params={"index_type": "INVERTED"})
  # hello_milvus.create_index("string1", index_params={"index_type": "TRIE"})
  #hello_milvus.create_index("int8_1", index_params={"index_type": "INVERTED", "mmap.enabled": True})
  #print("create scalar index done")
  hello_milvus.create_index("embeddings", index_hnsw)
  #hello_milvus.compact()
  print("xxinsert and flush done")

def load_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  #hello_milvus.release()
  #hello_milvus.drop_index(index_name="string1")
  #hello_milvus.create_index("string1", index_params={"index_type": "INVERTED"})
  time.sleep(10)
  hello_milvus.load()
  #hello_milvus.create_index("int1", "int1_index")
  print("load done")

def release_collection():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  #hello_milvus.release()
  #hello_milvus.create_index("pk")
  time.sleep(10)
  hello_milvus.release()
  #hello_milvus.create_index("int1", "int1_index")
  print("release done")

def search():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  vectors_to_search = [[random.random() for _ in range(128)] for _ in range(2)]
  search_params_hnsw = {
        "metric_type": "L2",
        "params": {"ef": 10},
    }
  result = hello_milvus.search(vectors_to_search, "embeddings", search_params_hnsw, limit=100, expr = "int1 in [10, 20]") 
  print(result)
  result = hello_milvus.search(vectors_to_search, "embeddings", search_params_hnsw, limit=100) 
  print(result)
  result = hello_milvus.search(vectors_to_search, "embeddings", search_params_hnsw, limit=100) 

  
def query():
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  
  # Create expression with all strings from the pool
  expr = f"string1 in {string_pool}"
  print (expr)
  result = hello_milvus.query(expr=expr, output_fields=["count(*)"])
  print(f"Query with string pool: {len(result)} results")
  print(result)


def build_in_expr(field: str, values: list[int]) -> str:
    if not values:
        raise ValueError("查询列表不能为空")
    values_str = ", ".join(str(v) for v in values)
    return f"{field} in [{values_str}]"

if __name__ == "__main__":
  #init_collection()
  #release_collection()
  load_collection()
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  while (1):
     # Create expression with all strings from the pool
     result = hello_milvus.query(expr=build_in_expr("int2", [i for i in range(1)]), output_fields=["count(*)"])
     print(result)
    #  print(len(string_pool))
    #  expr = f"string1 in {string_pool}"
    #  #print (expr)
    #  result = hello_milvus.query(expr=expr, output_fields=["count(*)"], expr_params={"use_set_for_term_in": "true"})
    #  print(f"Query with string pool: {len(result)} results")
    #  print(result)
  # while(1):
  #   concurrent_query(hello_milvus, thread_num=100)
  # while(1):
  #   query()

