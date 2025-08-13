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
from concurrent.futures import ThreadPoolExecutor

def generate_random_string():
    length = random.randint(1, 1000)
    characters = string.ascii_letters
    random_string = ''.join(random.choice(characters) for _ in range(1000))
    return random_string

str_prefix = generate_random_string()

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

def get_json(keys_num):
  dict1 = {f"key_{i}x": random_value() for i in range(1, keys_num)}
  dict1["key_0x"] = random.randint(1, 1000)
  dict1["keyxx"] = random.randint(1, 1000)
  return dict1
  #return json.dumps(dict1)

#   json['key_1'] == 1 && int1 == 0
def Test1(has_index):
  build_col = False
  fields = [
      FieldSchema("pk", dtype=DataType.INT64, is_primary=True),
      FieldSchema("int1", dtype=DataType.INT64),
      #FieldSchema("str1", dtype=DataType.VARCHAR, max_length=20000),
      FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=128)
  ]
  if (build_col):
    connections.connect("default", host="localhost", port="19530")
    print("connect done")
    utility.drop_collection("hello_milvus")
    print("drop table done")
  
    schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
    hello_milvus = Collection("hello_milvus", schema, shards_num=1)
    print("create table done")
    index_hnsw = {
        "index_type": "HNSW",
        "metric_type": "L2",
              "params": {"M": 8, "efConstruction" : 256},
    }
    hello_milvus.create_index("embeddings", index_hnsw)
    if (has_index):
      hello_milvus.create_index("str1", index_params={'index_type' : "BITMAP"})
    index = 0
    import random
    while(index < 10000):
      entities = [
          [i for i in range(100 *index, 100 * (index + 1))],  # field pk
          [int(random.randrange(0, 1000)) for _ in range(100)],  # field int1
          #[random.randint(0, 1000) for _ in range(100)],  # field int1
          #[ str(random.randint(0, 99)) for _ in range(100)],  # field random
          [[random.random() for _ in range(128)] for _ in range(100)],  # field embeddings
      ]
      #if index == 0:
      #  print(entities)
      insert_result = hello_milvus.insert(entities)
      index += 1
    hello_milvus.flush()
    print("insert and flush done")
    if (has_index):
      hello_milvus.create_index("str1", index_params={'index_type' : "BITMAP"})
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.release()
  hello_milvus.load()
  print("load done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  import random
  values = [random.randint(0, 99) for _ in range(20)]
  # cursor = 0
  # while (True):
  #   values = [random.randint(0, 99) for _ in range(5)]
  #   expr = " or ".join(f"str1 == '{v}'" for v in values)
  #   #expr = "str1 in [{}]".format(", ".join(f"'{v}'" for v in values))
  #   print(expr)
  #   result = hello_milvus.query(expr=expr, output_fields=["count(*)"])
  #   print(result)
  #   cursor += 1
  cursor = 0
  while (True):
    values = [random.randint(0, 99) for _ in range(10)]
    #expr = " or ".join(f"int1 == {v}" for v in values)
    expr =  "int1 in [{}]".format(', '.join(map(str, values)))
    print(expr)
    result = hello_milvus.query(expr=expr, output_fields=["count(*)"])
    print(result)
    cursor += 1

  
if __name__ == "__main__":
  Test1(False)

