from pymilvus import MilvusClient, DataType
import numpy as np
import random
from loguru import logger
import time
import json

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

def test_json_count_aggregation():
    """测试JSON计数聚合"""
    collection_name = "json_count_test"
    create_collection(collection_name)
    
    # 生成数据
    data = []
    for i in range(1000):
        count_json = {
            "category": random.choice(["A", "B", "C", "D", "E"]),
            "status": random.choice(["active", "inactive", "pending"]),
            "priority": random.randint(1, 5),
            "department": random.choice(["IT", "HR", "Finance", "Marketing", "Sales"]),
            "region": random.choice(["North", "South", "East", "West"]),
            "score": random.randint(0, 100),
            "tags": random.sample(["urgent", "important", "normal", "low", "critical"], random.randint(1, 3))
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": count_json
        })
    
    insert_data(collection_name, data)
    
    # 测试计数聚合
    count_queries = [
        "count(*)",  # 总计数
        "count(my_json['category'])",  # 非空category计数
        "count(my_json['score'])",  # 非空score计数
        "count(my_json['nonexistent'])",  # 不存在的字段计数
    ]
    
    for query in count_queries:
        logger.info(f"Testing count aggregation: {query}")
        try:
            res = client.query(collection_name=collection_name, filter="", output_fields=[query])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Count aggregation failed: {e}")

def test_json_conditional_count():
    """测试JSON条件计数"""
    collection_name = "json_conditional_count_test"
    create_collection(collection_name)
    
    # 生成数据
    data = []
    for i in range(1000):
        conditional_json = {
            "age": random.randint(18, 80),
            "salary": random.randint(30000, 150000),
            "experience": random.randint(0, 30),
            "education": random.choice(["high_school", "bachelor", "master", "phd"]),
            "location": random.choice(["urban", "suburban", "rural"]),
            "skills": random.sample(["python", "java", "javascript", "sql", "docker", "kubernetes"], random.randint(1, 4))
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": conditional_json
        })
    
    insert_data(collection_name, data)
    
    # 测试条件计数
    conditional_queries = [
        "my_json['age'] > 30",
        "my_json['salary'] > 80000",
        "my_json['education'] == 'master'",
        "my_json['location'] == 'urban'",
        "json_contains_any(my_json['skills'], ['python', 'java'])",
        "my_json['age'] > 30 and my_json['salary'] > 80000",
        "my_json['education'] == 'phd' or my_json['experience'] > 20",
        "my_json['age'] between 25 and 45 and my_json['location'] == 'urban'"
    ]
    
    for query in conditional_queries:
        logger.info(f"Testing conditional count: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Conditional count failed: {e}")

def test_json_group_by_analysis():
    """测试JSON分组分析"""
    collection_name = "json_group_by_test"
    create_collection(collection_name)
    
    # 生成数据
    data = []
    for i in range(2000):
        group_json = {
            "product_id": random.randint(1, 10),
            "category": random.choice(["electronics", "clothing", "books", "food", "sports"]),
            "brand": random.choice(["brand_A", "brand_B", "brand_C", "brand_D"]),
            "price": round(random.uniform(10.0, 1000.0), 2),
            "rating": round(random.uniform(1.0, 5.0), 1),
            "sales_count": random.randint(0, 1000),
            "region": random.choice(["US", "EU", "ASIA", "AFRICA"]),
            "season": random.choice(["spring", "summer", "autumn", "winter"])
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": group_json
        })
    
    insert_data(collection_name, data)
    
    # 测试分组分析（通过多次查询模拟）
    group_analyses = [
        # 按类别统计
        ("category", "electronics"),
        ("category", "clothing"),
        ("category", "books"),
        ("category", "food"),
        ("category", "sports"),
        
        # 按品牌统计
        ("brand", "brand_A"),
        ("brand", "brand_B"),
        ("brand", "brand_C"),
        ("brand", "brand_D"),
        
        # 按地区统计
        ("region", "US"),
        ("region", "EU"),
        ("region", "ASIA"),
        ("region", "AFRICA"),
        
        # 按季节统计
        ("season", "spring"),
        ("season", "summer"),
        ("season", "autumn"),
        ("season", "winter")
    ]
    
    for field, value in group_analyses:
        logger.info(f"Testing group by analysis: {field} = {value}")
        try:
            res = client.query(collection_name=collection_name, 
                              filter=f"my_json['{field}'] == '{value}'", 
                              output_fields=['count(*)'])
            logger.info(f"Count for {field}={value}: {res}")
        except Exception as e:
            logger.error(f"Group by analysis failed: {e}")

def test_json_statistical_queries():
    """测试JSON统计查询"""
    collection_name = "json_statistical_test"
    create_collection(collection_name)
    
    # 生成数据
    data = []
    for i in range(1000):
        stats_json = {
            "temperature": round(random.uniform(-20.0, 40.0), 1),
            "humidity": round(random.uniform(0.0, 100.0), 1),
            "pressure": round(random.uniform(900.0, 1100.0), 1),
            "wind_speed": round(random.uniform(0.0, 50.0), 1),
            "precipitation": round(random.uniform(0.0, 100.0), 1),
            "uv_index": random.randint(0, 11),
            "air_quality": random.randint(1, 500),
            "station_id": random.randint(1, 50),
            "timestamp": random.randint(1600000000, 1700000000)
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": stats_json
        })
    
    insert_data(collection_name, data)
    
    # 测试统计查询
    statistical_queries = [
        # 温度相关统计
        "my_json['temperature'] > 30",
        "my_json['temperature'] < 0",
        "my_json['temperature'] between 15 and 25",
        
        # 湿度相关统计
        "my_json['humidity'] > 80",
        "my_json['humidity'] < 30",
        "my_json['humidity'] between 40 and 60",
        
        # 空气质量相关统计
        "my_json['air_quality'] > 100",
        "my_json['air_quality'] < 50",
        "my_json['air_quality'] between 50 and 100",
        
        # 复合条件统计
        "my_json['temperature'] > 25 and my_json['humidity'] > 70",
        "my_json['air_quality'] > 150 and my_json['wind_speed'] < 10",
        "my_json['temperature'] between 20 and 30 and my_json['uv_index'] > 5",
        
        # 范围查询
        "my_json['pressure'] > 1000",
        "my_json['wind_speed'] > 20",
        "my_json['precipitation'] > 50"
    ]
    
    for query in statistical_queries:
        logger.info(f"Testing statistical query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Statistical query failed: {e}")

def test_json_time_series_analysis():
    """测试JSON时间序列分析"""
    collection_name = "json_timeseries_test"
    create_collection(collection_name)
    
    # 生成时间序列数据
    data = []
    base_timestamp = 1600000000
    for i in range(1000):
        # 生成连续的时间戳
        timestamp = base_timestamp + (i * 3600)  # 每小时一个数据点
        
        timeseries_json = {
            "timestamp": timestamp,
            "value": random.randint(100, 1000),
            "metric": random.choice(["cpu_usage", "memory_usage", "disk_usage", "network_traffic"]),
            "server_id": random.randint(1, 10),
            "status": random.choice(["normal", "warning", "critical"]),
            "response_time": round(random.uniform(10.0, 500.0), 2),
            "error_count": random.randint(0, 10),
            "user_count": random.randint(0, 1000)
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": timeseries_json
        })
    
    insert_data(collection_name, data)
    
    # 测试时间序列查询
    timeseries_queries = [
        # 时间范围查询
        "my_json['timestamp'] > 1600000000",
        "my_json['timestamp'] < 1600100000",
        "my_json['timestamp'] between 1600000000 and 1600100000",
        
        # 指标查询
        "my_json['metric'] == 'cpu_usage'",
        "my_json['metric'] == 'memory_usage'",
        "my_json['metric'] == 'disk_usage'",
        "my_json['metric'] == 'network_traffic'",
        
        # 状态查询
        "my_json['status'] == 'critical'",
        "my_json['status'] == 'warning'",
        "my_json['status'] == 'normal'",
        
        # 性能查询
        "my_json['response_time'] > 200",
        "my_json['error_count'] > 5",
        "my_json['user_count'] > 500",
        
        # 复合时间序列查询
        "my_json['metric'] == 'cpu_usage' and my_json['value'] > 800",
        "my_json['status'] == 'critical' and my_json['response_time'] > 300",
        "my_json['timestamp'] > 1600050000 and my_json['error_count'] > 3"
    ]
    
    for query in timeseries_queries:
        logger.info(f"Testing timeseries query: {query}")
        try:
            res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
            logger.info(f"Result: {res}")
        except Exception as e:
            logger.error(f"Timeseries query failed: {e}")

def test_json_performance_benchmark():
    """测试JSON性能基准"""
    collection_name = "json_benchmark_test"
    create_collection(collection_name)
    
    # 生成大量数据用于性能测试
    data = []
    for i in range(5000):
        benchmark_json = {
            "user_id": i,
            "session_id": f"session_{random.randint(1000, 9999)}",
            "page_id": random.randint(1, 100),
            "action_type": random.choice(["view", "click", "scroll", "submit", "download"]),
            "duration": random.randint(1, 3600),
            "timestamp": random.randint(1600000000, 1700000000),
            "device_type": random.choice(["desktop", "mobile", "tablet"]),
            "browser": random.choice(["chrome", "firefox", "safari", "edge"]),
            "country": random.choice(["US", "CN", "JP", "DE", "UK", "FR", "CA", "AU"]),
            "referrer": random.choice(["google", "direct", "facebook", "twitter", "linkedin"])
        }
        
        data.append({
            "my_id": i,
            "my_vector": [random.random() for _ in range(dim)],
            "my_json": benchmark_json
        })
    
    insert_data(collection_name, data)
    
    # 性能基准查询
    benchmark_queries = [
        "my_json['action_type'] == 'click'",
        "my_json['device_type'] == 'mobile'",
        "my_json['country'] == 'US'",
        "my_json['duration'] > 1800",
        "my_json['page_id'] in [1, 2, 3, 4, 5]",
        "my_json['browser'] == 'chrome' and my_json['device_type'] == 'desktop'",
        "my_json['action_type'] == 'view' and my_json['duration'] > 300",
        "my_json['country'] in ['US', 'CN', 'JP'] and my_json['action_type'] == 'click'"
    ]
    
    logger.info("Starting performance benchmark...")
    for query in benchmark_queries:
        logger.info(f"Benchmarking query: {query}")
        
        # 运行多次取平均值
        times = []
        for _ in range(5):
            try:
                start_time = time.time()
                res = client.query(collection_name=collection_name, filter=query, output_fields=['count(*)'])
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
            except Exception as e:
                logger.error(f"Benchmark query failed: {e}")
                break
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            logger.info(f"Average: {avg_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")

if __name__ == "__main__":
    logger.info("Starting JSON aggregation tests...")
    
    # 运行各种聚合测试
    test_json_count_aggregation()
    test_json_conditional_count()
    test_json_group_by_analysis()
    test_json_statistical_queries()
    test_json_time_series_analysis()
    test_json_performance_benchmark()
    
    logger.info("All JSON aggregation tests completed!") 