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
def Test1():
  build_col = True
  fields = [
      FieldSchema("pk", dtype=DataType.INT64, is_primary=True),
      FieldSchema("int1", dtype=DataType.INT64),
      FieldSchema("str1", dtype=DataType.VARCHAR, max_length=20000),
      FieldSchema("json1", dtype=DataType.JSON, max_length=20000), 
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
    import random
    index = 0
    while(index < 300):
      entities = [
          [i for i in range(100 *index, 100 * (index + 1))],  # field pk
          #[int(random.randrange(0, 4)) for _ in range(100)],  # field int1
          [i for i in range(100 *index, 100 * (index + 1))],  # field int1
          [ "xxx" + str(i%20) for i in range(100)],  # field random
          [ get_json(100) for i in range(100)],
          [[random.random() for _ in range(128)] for _ in range(100)],  # field embeddings
      ]
      #if index == 0:
      #  print(entities)
      insert_result = hello_milvus.insert(entities)
      index += 1
    hello_milvus.flush()
    print("insert and flush done")
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  result = hello_milvus.query(expr="exists json1['keyxx'] ", output_fields=["count(*)"])
  print(result)
  result = hello_milvus.query(expr=" 0 < int1 <= 100000", output_fields=["count(*)"])
  print (result)

#   str1 = 'xxx100' && str2 = 'xxx100' && str3 = 'xxx10000'
def Test2(build_index = False):
  build_col = True
  fields = [
      FieldSchema("pk", dtype=DataType.INT64, is_primary=True),
      FieldSchema("str1", dtype=DataType.VARCHAR, max_length=20000),
      FieldSchema("str2", dtype=DataType.VARCHAR, max_length=20000),
      FieldSchema("str3", dtype=DataType.VARCHAR, max_length=20000),
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
    if build_index:
      hello_milvus.create_index("str3", index_params={"index_type": "BITMAP"})
    import random
    index = 0
    while(index < 300):
      entities = [
          [i for i in range(100 *index, 100 * (index + 1))],  # field pk
          [ "xxx" + str(i%100) for i in range(100)],  # field str1
          [ "xxx" + str(i%100) for i in range(100)],  # field str1
          [ "xxx" + str(i%100) for i in range(100)],  # field str1
          [[random.random() for _ in range(128)] for _ in range(100)],  # field embeddings
      ]
      #if index == 0:
      #  print(entities)
      insert_result = hello_milvus.insert(entities)
      index += 1
    hello_milvus.flush()
    if build_index:
      hello_milvus.create_index("str3", index_params={"index_type": "BITMAP"})
    print("insert and flush done")
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  result = hello_milvus.query(expr=" str1 == 'xxx1' || str2 == 'xxx5' || str3 == 'xxx10'", output_fields=["pk"])
  print(result)
  result = hello_milvus.query(expr=" str1 == 'xxx1'", output_fields=["count(*)"])
  print (result)

#json['key_1'] == 1 && str1 == "xx99"
def Test3():
  build_col = True
  fields = [
      FieldSchema("pk", dtype=DataType.INT64, is_primary=True),
      FieldSchema("int1", dtype=DataType.INT64),
      FieldSchema("str1", dtype=DataType.VARCHAR, max_length=20000),
      FieldSchema("json1", dtype=DataType.JSON, max_length=20000), 
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
    hello_milvus.create_index("str1", index_params={"index_type": "INVERTED"})
    import random
    index = 0
    while(index < 300):
      entities = [
          [i for i in range(100 *index, 100 * (index + 1))],  # field pk
          #[int(random.randrange(0, 4)) for _ in range(100)],  # field int1
          [i for i in range(100 *index, 100 * (index + 1))],  # field int1
          [ "xxx" + str(i%1000) for i in range(100)],  # field random
          [ get_json(100) for i in range(100)],
          [[random.random() for _ in range(128)] for _ in range(100)],  # field embeddings
      ]
      #if index == 0:
      #  print(entities)
      insert_result = hello_milvus.insert(entities)
      index += 1
    hello_milvus.flush()
    hello_milvus.create_index("str1", index_params={"index_type": "INVERTED"})
    print("insert and flush done")
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  result = hello_milvus.query(expr="json1['key_1'] == 1  && str1 == 'xxx10'", output_fields=["pk"])
  print(result)
  result = hello_milvus.query(expr=" str1 == 'xxx10' ", output_fields=["count(*)"])
  print (result)

#json['key_0'] == 1 or json['key_0'] == 5 or json['key_0'] == 50
def Test4():
  build_col = True
  fields = [
      FieldSchema("pk", dtype=DataType.INT64, is_primary=True),
      FieldSchema("json1", dtype=DataType.JSON, max_length=20000), 
      FieldSchema("json2", dtype=DataType.JSON, max_length=20000), 
      FieldSchema("json3", dtype=DataType.JSON, max_length=20000), 
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
    import random
    index = 0
    while(index < 30):
      entities = [
          [i for i in range(100 *index, 100 * (index + 1))],  # field pk
          [ get_json(100) for i in range(100)],
          [ get_json(100) for i in range(100)],
          [ get_json(100) for i in range(100)],
          [[random.random() for _ in range(128)] for _ in range(100)],  # field embeddings
      ]
      #if index == 0:
      #  print(entities)
      insert_result = hello_milvus.insert(entities)
      index += 1
    hello_milvus.flush()
    print("insert and flush done")
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  result = hello_milvus.query(expr="json1['key_0'] == 1 or json1['key_0'] \
                              == 5 or json1['key_0'] == 50", output_fields=["pk"])
  print(result)
  result = hello_milvus.query(expr="json1['key_0'] == 215", output_fields=["json1"])
  print (result)


def generate_random(total, sample):
    x = random.randint(0, total - 1) % sample
    return x

def generate_int_array(total, sample, size):
  result = []
  for _ in range(size):
    result.append(generate_random(total, sample))
  return result

def generate_varchar_array(total, sample, size):
  result = []
  for _ in range(size):
    result.append("xxx" + str(generate_random(total, sample)))
  return result

def Test5():
  build_col = True
  fields = [
      FieldSchema("pk", dtype=DataType.INT64, is_primary=True),
      FieldSchema("int1", dtype=DataType.INT64),
      FieldSchema("str1", dtype=DataType.VARCHAR, max_length=2000),
      FieldSchema("str2", dtype=DataType.VARCHAR, max_length=2000),
      FieldSchema("int_array", dtype=DataType.ARRAY,  max_capacity=128, element_type=DataType.INT64),
      FieldSchema(name="varchar_array", dtype=DataType.ARRAY, max_capacity=128,
                element_type=DataType.VARCHAR, max_length=1000),
      FieldSchema("json1", dtype=DataType.JSON, max_length=20000), 
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
    import random
    index = 0
    while(index < 300):
      entities = [
          [i for i in range(100 *index, 100 * (index + 1))],  # field pk
          [i for i in range(100 *index, 100 * (index + 1))],  # int1
          [ "xxx" + str(i%100) for i in range(100)],  # field str1
          [ "xxx" + str(i%100) for i in range(100)],  # field str2
          [generate_int_array(100, 100, 50) for i in range(100)], # int_array,
          [generate_varchar_array(100, 100, 50) for i in range(100)], # int_array,
          [ get_json(100) for i in range(100)], 
          [[random.random() for _ in range(128)] for _ in range(100)],  # field embeddings
      ]
      #if index == 0:
      #  print(entities)
      insert_result = hello_milvus.insert(entities)
      index += 1
    hello_milvus.flush()
    print("insert and flush done")
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  #hello_milvus.create_index("str2", index_params={"index_type": "INVERTED"})
  hello_milvus.load()
  print("load done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  #result = hello_milvus.query(expr=' str1== "xxx10" && str2 == "xxx10" && int1 > 10 && json1["keyxx"] >200  ', 
                              #output_fields=["count(*)"])
  #print(result)
  result = hello_milvus.query(expr=" array_contains(int_array, 10) && ( str1== 'xxx10' || str2 == 'xxx1') && int1 > 100  ", 
                              output_fields=["count(*)"])
  print(result)

  #json['key_1'] == 1 && pk == "99"
def Test6():
  build_col = True
  fields = [
      FieldSchema("pk", dtype=DataType.INT64, is_primary=True),
      FieldSchema("int1", dtype=DataType.INT64),
      FieldSchema("str1", dtype=DataType.VARCHAR, max_length=20000),
      FieldSchema("json1", dtype=DataType.JSON, max_length=20000), 
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
    hello_milvus.create_index("str1", index_params={"index_type": "INVERTED"})
    import random
    index = 0
    while(index < 300):
      entities = [
          [i for i in range(100 *index, 100 * (index + 1))],  # field pk
          #[int(random.randrange(0, 4)) for _ in range(100)],  # field int1
          [i for i in range(100 *index, 100 * (index + 1))],  # field int1
          [ "xxx" + str(i%1000) for i in range(100)],  # field random
          [ get_json(100) for i in range(100)],
          [[random.random() for _ in range(128)] for _ in range(100)],  # field embeddings
      ]
      #if index == 0:
      #  print(entities)
      insert_result = hello_milvus.insert(entities)
      index += 1
    hello_milvus.flush()
    hello_milvus.create_index("str1", index_params={"index_type": "INVERTED"})
    print("insert and flush done")
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  result = hello_milvus.query(expr="json1['key_1'] == 1  && pk ==99", output_fields=["pk"])
  print(result)

  
def Test_NULL():
  build_col = True
  fields = [
      FieldSchema("pk", dtype=DataType.INT64, is_primary=True),
      FieldSchema("int1", dtype=DataType.INT64),
      FieldSchema("str1", dtype=DataType.VARCHAR, max_length=20000, nullable=True),
      FieldSchema("json1", dtype=DataType.JSON, max_length=20000), 
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
    #hello_milvus.create_index("str1", index_params={"index_type": "INVERTED"})
    import random
    index = 0
    while(index < 300):
      entities = [
          [i for i in range(100 *index, 100 * (index + 1))],  # field pk
          #[int(random.randrange(0, 4)) for _ in range(100)],  # field int1
          [i for i in range(100 *index, 100 * (index + 1))],  # field int1
          [ "xxx" + str(i) if i % 100 < 10  else None for i in range(100)],  # field random
          [ get_json(100) for i in range(100)],
          [[random.random() for _ in range(128)] for _ in range(100)],  # field embeddings
      ]

      #if index == 0:
      #  print(entities)
      insert_result = hello_milvus.insert(entities)
      index += 1
    hello_milvus.flush()
    #hello_milvus.create_index("str1", index_params={"index_type": "INVERTED"})
    print("insert and flush done")
  connections.connect("default", host="localhost", port="19530")
  print("connect done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  hello_milvus.load()
  print("load done")
  schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
  hello_milvus = Collection("hello_milvus", schema)
  result = hello_milvus.query(expr="int1 % 8192  < 5000 && str1 < 'xxxxxx'", output_fields=["pk"])
  print(result)
   
if __name__ == "__main__":
  Test1()
  Test2()
  Test2(True)
  Test3()
  Test4()
  Test5()
  Test6()
  Test_NULL()

