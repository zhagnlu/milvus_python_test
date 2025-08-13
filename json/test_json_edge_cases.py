from pymilvus import MilvusClient, DataType
import numpy as np
import random
from loguru import logger
import time
import json
import sys

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

def test_json_extreme_values():
    """测试JSON极值"""
    collection_name = "json_extreme_test"
    create_collection(collection_name)
    
    # 生成包含极值的JSON数据
    data = []
    for i in range(100):
        extreme_json = {
            "max_int": 2**63 - 1,  # 最大64位整数
            "min_int": -2**63,     # 最小64位整数
            "max_float": sys.float_info.max,
            "min_float": sys.float_info.min,
            "infinity": float('inf'),
            "negative_infinity": float('-inf'),
            "nan": float('nan'),
            "very_long_string": "x" * 10000,  # 超长字符串
            "empty_array": [],
            "empty_object": {},
            "deep_nested": {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "level5": {
                                    "value": i
                                }
                            }
                        }
                    }
                }
            }
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": extreme_json
        })
    
    insert_data(collection_name, data)
    
    # 测试极值查询
    queries = [
        "my_json['max_int'] == 9223372036854775807",
        "my_json['min_int'] == -9223372036854775808",
        "my_json['very_long_string'] like 'x%'",
        "json_length(my_json['empty_array']) == 0",
        "my_json['deep_nested']['level1']['level2']['level3']['level4']['level5']['value'] > 50"
    ]
    
    for query in queries:
        logger.info(f"Testing extreme value query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Extreme value query failed: {e}")

def test_json_special_characters():
    """测试JSON特殊字符"""
    collection_name = "json_special_chars_test"
    create_collection(collection_name)
    
    # 生成包含特殊字符的JSON数据
    special_chars = [
        "normal", "with spaces", "with-dashes", "with_underscores",
        "with.dots", "with/slashes", "with\\backslashes", "with:colons",
        "with;semicolons", "with,commas", "with\"quotes", "with'apostrophes",
        "with(brackets)", "with[braces]", "with{curly}", "with<angles>",
        "with&ampersands", "with|pipes", "with!exclamation", "with?question",
        "with@at", "with#hash", "with$dollar", "with%percent", "with^caret",
        "with*asterisk", "with+plus", "with=equals", "with~tilde", "with`backtick"
    ]
    
    data = []
    for i in range(100):
        special_json = {
            "normal_key": f"value_{i}",
            "key with spaces": f"value with spaces {i}",
            "key-with-dashes": f"value-with-dashes-{i}",
            "key_with_underscores": f"value_with_underscores_{i}",
            "key.with.dots": f"value.with.dots.{i}",
            "key/with/slashes": f"value/with/slashes/{i}",
            "key\\with\\backslashes": f"value\\with\\backslashes\\{i}",
            "key:with:colons": f"value:with:colons:{i}",
            "key;with;semicolons": f"value;with;semicolons;{i}",
            "key,with,commas": f"value,with,commas,{i}",
            "key\"with\"quotes": f"value\"with\"quotes\"{i}",
            "key'with'apostrophes": f"value'with'apostrophes'{i}",
            "key(with)brackets": f"value(with)brackets{i}",
            "key[with]braces": f"value[with]braces{i}",
            "key{with}curly": f"value{{with}}curly{i}",
            "key<with>angles": f"value<with>angles{i}",
            "key&with&ampersands": f"value&with&ampersands{i}",
            "key|with|pipes": f"value|with|pipes{i}",
            "key!with!exclamation": f"value!with!exclamation{i}",
            "key?with?question": f"value?with?question{i}",
            "key@with@at": f"value@with@at{i}",
            "key#with#hash": f"value#with#hash{i}",
            "key$with$dollar": f"value$with$dollar{i}",
            "key%with%percent": f"value%with%percent{i}",
            "key^with^caret": f"value^with^caret{i}",
            "key*with*asterisk": f"value*with*asterisk{i}",
            "key+with+plus": f"value+with+plus{i}",
            "key=with=equals": f"value=with=equals{i}",
            "key~with~tilde": f"value~with~tilde{i}",
            "key`with`backtick": f"value`with`backtick{i}"
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": special_json
        })
    
    insert_data(collection_name, data)
    
    # 测试特殊字符查询
    queries = [
        "my_json['normal_key'] == 'value_0'",
        "my_json['key with spaces'] like 'value with spaces%'",
        "my_json['key-with-dashes'] == 'value-with-dashes-0'",
        "my_json['key_with_underscores'] == 'value_with_underscores_0'",
        "my_json['key.with.dots'] == 'value.with.dots.0'"
    ]
    
    for query in queries:
        logger.info(f"Testing special char query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Special char query failed: {e}")

def test_json_unicode():
    """测试JSON Unicode字符"""
    collection_name = "json_unicode_test"
    create_collection(collection_name)
    
    # 生成包含Unicode字符的JSON数据
    unicode_strings = [
        "Hello World",  # 英文
        "你好世界",      # 中文
        "こんにちは世界",  # 日文
        "안녕하세요 세계",  # 韩文
        "Привет мир",   # 俄文
        "Hola mundo",   # 西班牙文
        "Bonjour le monde",  # 法文
        "Hallo Welt",   # 德文
        "Ciao mondo",   # 意大利文
        "Olá mundo",    # 葡萄牙文
        "مرحبا بالعالم",  # 阿拉伯文
        "नमस्ते दुनिया",  # 印地文
        "สวัสดีชาวโลก",   # 泰文
        "Xin chào thế giới",  # 越南文
        "Γεια σου κόσμε",  # 希腊文
        "שלום עולם",     # 希伯来文
        "Merhaba dünya",  # 土耳其文
        "Witaj świecie",  # 波兰文
        "Hej världen",   # 瑞典文
        "Hallo verden"   # 挪威文
    ]
    
    data = []
    for i in range(100):
        unicode_json = {
            "english": "Hello World",
            "chinese": "你好世界",
            "japanese": "こんにちは世界",
            "korean": "안녕하세요 세계",
            "russian": "Привет мир",
            "spanish": "Hola mundo",
            "french": "Bonjour le monde",
            "german": "Hallo Welt",
            "italian": "Ciao mondo",
            "portuguese": "Olá mundo",
            "arabic": "مرحبا بالعالم",
            "hindi": "नमस्ते दुनिया",
            "thai": "สวัสดีชาวโลก",
            "vietnamese": "Xin chào thế giới",
            "greek": "Γεια σου κόσμε",
            "hebrew": "שלום עולם",
            "turkish": "Merhaba dünya",
            "polish": "Witaj świecie",
            "swedish": "Hej världen",
            "norwegian": "Hallo verden",
            "mixed": f"{unicode_strings[i % len(unicode_strings)]}_{i}",
            "emoji": "🚀🌟🎉💯🔥",
            "special_unicode": "café naïve naïve résumé"
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": unicode_json
        })
    
    insert_data(collection_name, data)
    
    # 测试Unicode查询
    queries = [
        "my_json['english'] == 'Hello World'",
        "my_json['chinese'] == '你好世界'",
        "my_json['japanese'] == 'こんにちは世界'",
        "my_json['emoji'] == '🚀🌟🎉💯🔥'",
        "my_json['mixed'] like '%你好%'",
        "my_json['special_unicode'] == 'café naïve naïve résumé'"
    ]
    
    for query in queries:
        logger.info(f"Testing unicode query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Unicode query failed: {e}")

def test_json_large_objects():
    """测试大型JSON对象"""
    collection_name = "json_large_test"
    create_collection(collection_name)
    
    # 生成大型JSON对象
    data = []
    for i in range(50):  # 减少数量，因为对象很大
        large_json = {
            "id": i,
            "large_array": [random.randint(1, 1000) for _ in range(1000)],
            "large_string": "x" * 50000,  # 50KB字符串
            "nested_objects": {
                f"level1_{j}": {
                    f"level2_{k}": {
                        f"level3_{l}": {
                            "value": random.randint(1, 100),
                            "text": f"text_{i}_{j}_{k}_{l}"
                        }
                        for l in range(5)
                    }
                    for k in range(10)
                }
                for j in range(20)
            },
            "mixed_data": {
                f"key_{j}": random.choice([
                    random.randint(1, 1000),
                    f"string_{j}",
                    random.choice([True, False]),
                    [random.randint(1, 100) for _ in range(10)],
                    {"nested": random.randint(1, 100)}
                ])
                for j in range(100)
            }
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": large_json
        })
    
    insert_data(collection_name, data)
    
    # 测试大型对象查询
    queries = [
        "my_json['id'] == 0",
        "json_length(my_json['large_array']) == 1000",
        "my_json['large_array'][0] > 500",
        "my_json['nested_objects']['level1_0']['level2_0']['level3_0']['value'] > 50",
        "my_json['mixed_data']['key_0'] > 500"
    ]
    
    for query in queries:
        logger.info(f"Testing large object query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Large object query failed: {e}")

def test_json_malformed_queries():
    """测试格式错误的JSON查询"""
    collection_name = "json_malformed_test"
    create_collection(collection_name)
    
    # 插入一些简单数据
    data = []
    for i in range(100):
        simple_json = {
            "id": i,
            "name": f"user_{i}",
            "age": random.randint(18, 80),
            "active": random.choice([True, False])
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": simple_json
        })
    
    insert_data(collection_name, data)
    
    # 测试格式错误的查询
    malformed_queries = [
        "my_json['nonexistent_key'] == 1",  # 不存在的键
        "my_json['id']['nonexistent_nested'] == 1",  # 在非对象上访问嵌套键
        "my_json['large_array'][9999] == 1",  # 数组越界
        "my_json['id'] == 'not_a_number'",  # 类型不匹配
        "my_json['id'] > 'string_value'",  # 类型不匹配的比较
        "my_json['id'] in [1, 'string', 3]",  # 混合类型的IN查询
        "my_json['id'] between 'a' and 'z'",  # 字符串的BETWEEN查询
        "my_json['id'] like 'pattern'",  # 数字的LIKE查询
        "json_contains(my_json['id'], 1)",  # 在非数组上使用json_contains
        "json_length(my_json['id'])",  # 在非数组上使用json_length
        "my_json['id'] == null",  # 使用null而不是is null
        "my_json['id'] == undefined",  # 使用undefined
        "my_json['id'] == true and my_json['id'] == false",  # 矛盾的条件
        "my_json['id'] > 1000 or my_json['id'] < 0",  # 可能为真的条件
    ]
    
    for query in malformed_queries:
        logger.info(f"Testing malformed query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.info(f"Expected error for malformed query: {e}")

def test_json_concurrent_operations():
    """测试JSON并发操作"""
    collection_name = "json_concurrent_test"
    create_collection(collection_name)
    
    # 插入数据
    data = []
    for i in range(1000):
        concurrent_json = {
            "user_id": i,
            "session_id": f"session_{random.randint(1000, 9999)}",
            "timestamp": random.randint(1600000000, 1700000000),
            "action": random.choice(["read", "write", "delete", "update"]),
            "data": {
                "value": random.randint(1, 1000),
                "status": random.choice(["pending", "completed", "failed"]),
                "priority": random.randint(1, 10)
            }
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": concurrent_json
        })
    
    insert_data(collection_name, data)
    
    # 并发查询测试
    concurrent_queries = [
        "my_json['action'] == 'read'",
        "my_json['action'] == 'write'",
        "my_json['action'] == 'delete'",
        "my_json['action'] == 'update'",
        "my_json['data']['status'] == 'pending'",
        "my_json['data']['status'] == 'completed'",
        "my_json['data']['status'] == 'failed'",
        "my_json['data']['priority'] > 5",
        "my_json['timestamp'] > 1650000000",
        "my_json['data']['value'] > 500"
    ]
    
    logger.info("Testing concurrent queries...")
    for i in range(10):  # 运行10轮并发测试
        logger.info(f"Concurrent test round {i+1}")
        for query in concurrent_queries:
            try:
                start_time = time.time()
                res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
                end_time = time.time()
                logger.info(f"Query: {query[:50]}... took {(end_time - start_time) * 1000:.2f}ms")
            except Exception as e:
                logger.error(f"Concurrent query failed: {e}")

if __name__ == "__main__":
    logger.info("Starting JSON edge case tests...")
    
    # 运行各种边界情况测试
    test_json_extreme_values()
    test_json_special_characters()
    test_json_unicode()
    test_json_large_objects()
    test_json_malformed_queries()
    test_json_concurrent_operations()
    
    logger.info("All JSON edge case tests completed!") 