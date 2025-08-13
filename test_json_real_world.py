from pymilvus import MilvusClient, DataType
import numpy as np
import random
from loguru import logger
import time
import json
from enhanced_json_generator import EnhancedJSONGenerator

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

def test_ecommerce_scenario():
    """测试电商场景"""
    collection_name = "ecommerce_test"
    create_collection(collection_name)
    
    generator = EnhancedJSONGenerator()
    
    # 生成电商数据
    data = []
    for i in range(1000):
        # 混合生成用户档案、产品和订单数据
        if i % 3 == 0:
            json_data = generator.random_user_profile(i)
        elif i % 3 == 1:
            json_data = generator.random_product_data(i)
        else:
            json_data = generator.random_order_data(i)
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": json_data
        })
    
    insert_data(collection_name, data)
    
    # 电商相关查询
    ecommerce_queries = [
        # 用户相关查询
        "my_json['personal_info']['gender'] == 'female'",
        "my_json['personal_info']['address']['state'] == 'CA'",
        "my_json['preferences']['subscription_tier'] == 'premium'",
        "my_json['account']['status'] == 'active'",
        
        # 产品相关查询
        "my_json['basic_info']['category'] == 'electronics'",
        "my_json['pricing']['base_price'] > 500",
        "my_json['inventory']['stock_quantity'] > 100",
        "my_json['ratings']['average_rating'] > 4.0",
        "json_contains_any(my_json['specifications']['features'], ['wireless', 'bluetooth'])",
        
        # 订单相关查询
        "my_json['order_details']['status'] == 'delivered'",
        "my_json['order_details']['payment_status'] == 'paid'",
        "my_json['totals']['total'] > 1000",
        "json_length(my_json['items']) > 2",
        
        # 复合查询
        "my_json['personal_info']['address']['state'] == 'NY' and my_json['account']['status'] == 'active'",
        "my_json['basic_info']['category'] == 'electronics' and my_json['pricing']['base_price'] > 300",
        "my_json['order_details']['status'] == 'shipped' and my_json['totals']['total'] > 500"
    ]
    
    for query in ecommerce_queries:
        logger.info(f"Testing ecommerce query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Ecommerce query failed: {e}")

def test_logging_scenario():
    """测试日志分析场景"""
    collection_name = "logging_test"
    create_collection(collection_name)
    
    generator = EnhancedJSONGenerator()
    
    # 生成日志数据
    data = []
    for i in range(1000):
        json_data = generator.random_log_data(i)
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": json_data
        })
    
    insert_data(collection_name, data)
    
    # 日志分析查询
    logging_queries = [
        # 日志级别查询
        "my_json['level'] == 'ERROR'",
        "my_json['level'] == 'CRITICAL'",
        "my_json['level'] in ['WARNING', 'ERROR', 'CRITICAL']",
        
        # 服务查询
        "my_json['service'] == 'database'",
        "my_json['service'] == 'auth-service'",
        "my_json['service'] in ['web-server', 'database']",
        
        # 性能查询
        "my_json['performance']['response_time_ms'] > 1000",
        "my_json['performance']['memory_usage_mb'] > 500",
        "my_json['performance']['cpu_usage_percent'] > 80",
        
        # 错误查询
        "my_json['error_details']['error_code'] >= 500",
        "my_json['error_details']['error_type'] == 'DatabaseError'",
        "my_json['error_details']['recovered'] == false",
        
        # 复合查询
        "my_json['level'] == 'ERROR' and my_json['performance']['response_time_ms'] > 2000",
        "my_json['service'] == 'database' and my_json['performance']['cpu_usage_percent'] > 90",
        "my_json['error_details']['error_code'] >= 400 and my_json['error_details']['recovered'] == false"
    ]
    
    for query in logging_queries:
        logger.info(f"Testing logging query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Logging query failed: {e}")

def test_analytics_scenario():
    """测试分析场景"""
    collection_name = "analytics_test"
    create_collection(collection_name)
    
    generator = EnhancedJSONGenerator()
    
    # 生成分析数据
    data = []
    for i in range(1000):
        json_data = generator.random_analytics_data(i)
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": json_data
        })
    
    insert_data(collection_name, data)
    
    # 分析查询
    analytics_queries = [
        # 事件类型查询
        "my_json['event_type'] == 'purchase'",
        "my_json['event_type'] == 'page_view'",
        "my_json['event_type'] in ['purchase', 'download']",
        
        # 用户设备查询
        "my_json['user']['device_type'] == 'mobile'",
        "my_json['user']['browser'] == 'chrome'",
        "my_json['user']['os'] == 'ios'",
        
        # 地理位置查询
        "my_json['user']['country'] == 'US'",
        "my_json['user']['city'] == 'New York'",
        "my_json['user']['country'] in ['US', 'CN', 'JP']",
        
        # 性能查询
        "my_json['page']['load_time_ms'] > 3000",
        "my_json['interaction']['scroll_depth'] > 50",
        
        # 转化查询
        "my_json['conversion']['goal_completed'] == true",
        "my_json['conversion']['revenue'] > 100",
        "my_json['conversion']['funnel_step'] == 5",
        
        # 复合查询
        "my_json['event_type'] == 'purchase' and my_json['user']['device_type'] == 'desktop'",
        "my_json['conversion']['goal_completed'] == true and my_json['conversion']['revenue'] > 500",
        "my_json['user']['country'] == 'US' and my_json['page']['load_time_ms'] < 2000"
    ]
    
    for query in analytics_queries:
        logger.info(f"Testing analytics query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Analytics query failed: {e}")

def test_mixed_scenario():
    """测试混合场景"""
    collection_name = "mixed_test"
    create_collection(collection_name)
    
    generator = EnhancedJSONGenerator()
    
    # 生成混合数据
    data = []
    for i in range(2000):
        # 随机选择数据类型
        data_types = ["user_profile", "product", "order", "log", "analytics"]
        data_type = random.choice(data_types)
        
        if data_type == "user_profile":
            json_data = generator.random_user_profile(i)
        elif data_type == "product":
            json_data = generator.random_product_data(i)
        elif data_type == "order":
            json_data = generator.random_order_data(i)
        elif data_type == "log":
            json_data = generator.random_log_data(i)
        else:
            json_data = generator.random_analytics_data(i)
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": json_data
        })
    
    insert_data(collection_name, data)
    
    # 混合场景查询
    mixed_queries = [
        # 跨数据类型的通用查询
        "my_json['user_id'] > 5000",
        "my_json['timestamp'] > '2024-06-01'",
        "my_json['status'] == 'active'",
        
        # 嵌套字段查询
        "my_json['personal_info']['address']['state'] == 'CA'",
        "my_json['basic_info']['category'] == 'electronics'",
        "my_json['order_details']['payment_status'] == 'paid'",
        "my_json['performance']['response_time_ms'] > 1000",
        "my_json['user']['country'] == 'US'",
        
        # 数组查询
        "json_contains_any(my_json['specifications']['features'], ['wireless'])",
        "json_length(my_json['items']) > 1",
        
        # 复杂条件查询
        "my_json['personal_info']['address']['state'] == 'NY' and my_json['account']['status'] == 'active'",
        "my_json['basic_info']['category'] == 'electronics' and my_json['pricing']['base_price'] > 200",
        "my_json['level'] == 'ERROR' and my_json['performance']['response_time_ms'] > 2000",
        "my_json['event_type'] == 'purchase' and my_json['conversion']['revenue'] > 100"
    ]
    
    for query in mixed_queries:
        logger.info(f"Testing mixed query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Mixed query failed: {e}")

def test_performance_comparison():
    """测试不同数据类型的性能对比"""
    generator = EnhancedJSONGenerator()
    
    # 测试不同数据类型的性能
    data_types = ["user_profile", "product", "order", "log", "analytics"]
    
    for data_type in data_types:
        collection_name = f"perf_{data_type}_test"
        create_collection(collection_name)
        
        logger.info(f"Testing performance for {data_type}")
        
        # 生成数据
        data = []
        for i in range(1000):
            if data_type == "user_profile":
                json_data = generator.random_user_profile(i)
            elif data_type == "product":
                json_data = generator.random_product_data(i)
            elif data_type == "order":
                json_data = generator.random_order_data(i)
            elif data_type == "log":
                json_data = generator.random_log_data(i)
            else:
                json_data = generator.random_analytics_data(i)
            
            data.append({
                "my_id": i,
                "my_vector": [random.random() for _ in range(dim)],
                "my_json": json_data
            })
        
        insert_data(collection_name, data)
        
        # 性能测试查询
        if data_type == "user_profile":
            test_query = "my_json['personal_info']['address']['state'] == 'CA'"
        elif data_type == "product":
            test_query = "my_json['basic_info']['category'] == 'electronics'"
        elif data_type == "order":
            test_query = "my_json['order_details']['status'] == 'delivered'"
        elif data_type == "log":
            test_query = "my_json['level'] == 'ERROR'"
        else:  # analytics
            test_query = "my_json['event_type'] == 'purchase'"
        
        # 运行性能测试
        times = []
        for _ in range(10):
            try:
                start_time = time.time()
                res = client.query(collection_name=collection_name, filter=test_query, output_fields=['count(*)'])
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            except Exception as e:
                logger.error(f"Performance test failed: {e}")
                break
        
        if times:
            avg_time = sum(times) / len(times)
            logger.info(f"{data_type} average query time: {avg_time:.2f}ms")

if __name__ == "__main__":
    logger.info("Starting real-world JSON tests...")
    
    # 运行各种真实世界场景测试
    test_ecommerce_scenario()
    test_logging_scenario()
    test_analytics_scenario()
    test_mixed_scenario()
    test_performance_comparison()
    
    logger.info("All real-world JSON tests completed!") 