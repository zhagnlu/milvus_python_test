from pymilvus import MilvusClient, DataType
import numpy as np
import random
from loguru import logger
import time
import json
import random_json

client = MilvusClient()
logger.info("connected")

dim = 128
logger.info("creating index params")
schema = client.create_schema(enable_dynamic_field=True)
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="my_json", datatype=DataType.JSON)
logger.info("created schema")

def create_collection(collection_name):
    """创建集合"""
    res = client.list_collections()
    if collection_name in res:
        client.drop_collection(collection_name)
    
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"created collection: {collection_name}")

def insert_data(collection_name, data):
    """插入数据"""
    res = client.insert(collection_name=collection_name, data=data)
    logger.info(f"Inserted {len(data)} records")
    client.flush(collection_name=collection_name)
    client.load_collection(collection_name=collection_name)

def test_json_nested_queries():
    """测试嵌套JSON查询"""
    collection_name = "json_nested_test"
    create_collection(collection_name)
    
    # 生成嵌套JSON数据
    data = []
    for i in range(1000):
        nested_json = {
            "user": {
                "id": i,
                "profile": {
                    "name": f"user_{i}",
                    "age": random.randint(18, 80),
                    "preferences": {
                        "theme": random.choice(["dark", "light"]),
                        "language": random.choice(["en", "zh", "es"]),
                        "notifications": random.choice([True, False])
                    }
                },
                "scores": [random.randint(60, 100) for _ in range(5)],
                "tags": random.sample(["student", "teacher", "admin", "guest"], random.randint(1, 3))
            },
            "metadata": {
                "created_at": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "version": random.randint(1, 10),
                "active": random.choice([True, False])
            }
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": nested_json
        })
    
    insert_data(collection_name, data)
    
    # 测试各种嵌套查询
    queries = [
        "my_json['user']['profile']['age'] > 30",
        "my_json['user']['profile']['preferences']['theme'] == 'dark'",
        "my_json['metadata']['version'] >= 5",
        "my_json['user']['profile']['preferences']['notifications'] == true",
        "my_json['user']['id'] in [1, 2, 3, 4, 5]",
        "my_json['user']['profile']['name'] like 'user_1%'",
        "my_json['metadata']['active'] == false and my_json['user']['profile']['age'] < 50"
    ]
    
    for query in queries:
        logger.info(f"Testing query: {query}")
        try:
            while 1:
                use_stats_param = {"use_stats": "true"}
                not_use_stats_param = {"use_stats": "false"}
                res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'], filter_params=use_stats_param)
                logger.info(f"Result: {res}")
                res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'], filter_params=not_use_stats_param)
                logger.info(f"Result: {res}")
                time.sleep(1)
        except Exception as e:
            logger.error(f"Query failed: {e}")

def test_json_array_operations():
    """测试JSON数组操作"""
    collection_name = "json_array_test"
    create_collection(collection_name)
    
    # 生成包含数组的JSON数据
    data = []
    for i in range(1000):
        array_json = {
            "numbers": [random.randint(1, 100) for _ in range(random.randint(3, 8))],
            "strings": [f"item_{j}" for j in range(random.randint(2, 6))],
            "mixed": [random.choice([random.randint(1, 100), f"str_{j}", random.choice([True, False])]) 
                     for j in range(random.randint(3, 7))],
            "nested_arrays": [
                [random.randint(1, 10) for _ in range(3)],
                [f"nested_{j}" for j in range(2)]
            ],
            "objects_in_array": [
                {"id": j, "value": random.randint(1, 100)} 
                for j in range(random.randint(2, 5))
            ]
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": array_json
        })
    
    insert_data(collection_name, data)
    
    # 测试数组相关查询
    queries = [
        "json_contains(my_json['numbers'], 50)",
        "json_contains_any(my_json['strings'], ['item_0', 'item_1'])",
        "json_length(my_json['numbers']) > 5",
        "my_json['numbers'][0] > 50",
        "my_json['mixed'][1] == 'str_1'",
        "json_contains_all(my_json['strings'], ['item_0'])",
        "my_json['objects_in_array'][0]['value'] > 50"
    ]
    
    for query in queries:
        logger.info(f"Testing array query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Array query failed: {e}")

def test_json_type_operations():
    """测试JSON类型操作"""
    collection_name = "json_type_test"
    create_collection(collection_name)
    
    # 生成包含各种数据类型的JSON
    data = []
    for i in range(1000):
        type_json = {
            "integer": random.randint(-1000, 1000),
            "float": round(random.uniform(-1000.0, 1000.0), 2),
            "string": f"string_{i}",
            "boolean": random.choice([True, False]),
            "null_value": None,
            "big_int": 2**53 + random.randint(0, 100),
            "float64": float(np.float64(random.random())),
            "empty_string": "",
            "zero": 0,
            "negative": random.randint(-100, -1)
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": type_json
        })
    
    insert_data(collection_name, data)
    
    # 测试类型相关查询
    queries = [
        "my_json['integer'] > 0",
        "my_json['float'] between -500 and 500",
        "my_json['string'] like 'string_%'",
        "my_json['boolean'] == true",
        "my_json['null_value'] is null",
        "my_json['big_int'] > 9007199254740992",
        "my_json['empty_string'] == ''",
        "my_json['zero'] == 0",
        "my_json['negative'] < 0",
        "my_json['integer'] in [1, 2, 3, 4, 5]"
    ]
    
    for query in queries:
        logger.info(f"Testing type query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Type query failed: {e}")

def test_json_complex_conditions():
    """测试复杂JSON条件查询"""
    collection_name = "json_complex_test"
    create_collection(collection_name)
    
    # 生成复杂JSON数据
    data = []
    for i in range(1000):
        complex_json = {
            "category": random.choice(["A", "B", "C", "D"]),
            "score": random.randint(0, 100),
            "status": random.choice(["active", "inactive", "pending"]),
            "priority": random.randint(1, 5),
            "tags": random.sample(["urgent", "important", "normal", "low"], random.randint(1, 3)),
            "metadata": {
                "department": random.choice(["IT", "HR", "Finance", "Marketing"]),
                "location": random.choice(["NY", "LA", "SF", "CHI"]),
                "budget": random.randint(1000, 100000)
            }
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": complex_json
        })
    
    insert_data(collection_name, data)
    
    # 测试复杂条件查询
    queries = [
        "my_json['category'] == 'A' and my_json['score'] > 80",
        "my_json['status'] == 'active' or my_json['priority'] >= 4",
        "my_json['score'] between 60 and 90 and my_json['metadata']['budget'] > 50000",
        "my_json['category'] in ['A', 'B'] and my_json['metadata']['department'] == 'IT'",
        "json_contains_any(my_json['tags'], ['urgent', 'important']) and my_json['priority'] >= 3",
        "my_json['metadata']['location'] == 'NY' and my_json['score'] > 70 and my_json['status'] == 'active'",
        "not (my_json['status'] == 'inactive') and my_json['score'] >= 50"
    ]
    
    for query in queries:
        logger.info(f"Testing complex query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Complex query failed: {e}")



if __name__ == "__main__":
    logger.info("Starting comprehensive JSON tests...")
    
    # 运行各种测试
    test_json_nested_queries()
    # test_json_array_operations()
    # test_json_type_operations()
    # test_json_complex_conditions()

    
    logger.info("All JSON tests completed!") 