from pymilvus import MilvusClient, DataType
import numpy as np
import random
from loguru import logger
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class ConcurrentTest:
    def __init__(self, collection_name="concurrent_test"):
        self.client = MilvusClient()
        self.collection_name = collection_name
        self.dim = 128
        self.total_records = 1000000
        self.running = True
        self.lock = threading.Lock()
        self.wrong_count = 0
        
        logger.info("Connected to Milvus")
        
    def create_collection(self):
        """创建集合"""
        try:
            self.client.drop_collection(self.collection_name)
        except:
            pass
            
        schema = self.client.create_schema(enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="json_data", datatype=DataType.JSON)
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        
        self.client.create_collection(
            collection_name=self.collection_name, 
            schema=schema, 
            index_params=index_params
        )
        logger.info(f"Created collection: {self.collection_name}")
    
    def insert_initial_data(self):
        """插入初始100万条数据"""
        logger.info(f"Inserting {self.total_records} records...")
        
        batch_size = 10000
        total_batches = self.total_records // batch_size
        
        for batch_idx in range(total_batches):
            start_id = batch_idx * batch_size
            end_id = start_id + batch_size
            
            data = []
            for i in range(start_id, end_id):
                json_data = {
                    "user_id": i,
                    "score": random.uniform(0, 100),
                    "category": random.choice(["A", "B", "C", "D"]),
                    "active": random.choice([True, False])
                }
                
                record = {
                    "id": i,
                    "vector": [random.random() for _ in range(self.dim)],
                    "json_data": ""
                }
                data.append(record)
            
            self.client.insert(collection_name=self.collection_name, data=data)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Inserted batch {batch_idx + 1}/{total_batches}")
        
        self.client.flush(collection_name=self.collection_name)
        self.client.load_collection(collection_name=self.collection_name)
        logger.info("Initial data insertion completed")
    
    def upsert_worker(self):
        """持续upsert数据的线程"""
        logger.info("Upsert worker started")
        
        while self.running:
            try:
                num_to_update = random.randint(1000, 10000)
                ids_to_update = random.sample(range(self.total_records), num_to_update)
                
                data = []
                for record_id in ids_to_update:
                    json_data = {
                        "user_id": record_id,
                        "score": random.uniform(0, 100),
                        "category": random.choice(["A", "B", "C", "D"]),
                        "active": random.choice([True, False]),
                        "updated": True
                    }
                    
                    record = {
                        "id": record_id,
                        "vector": [random.random() for _ in range(self.dim)],
                        "json_data": ""
                    }
                    data.append(record)
                
                with self.lock:
                    self.client.upsert(collection_name=self.collection_name, data=data)
                
                logger.info(f"Upserted {num_to_update} records")
                time.sleep(random.uniform(0.1, 0.5))
                
            except Exception as e:
                logger.error(f"Error in upsert worker: {e}")
                time.sleep(1)
    
    def count_query_worker(self, worker_id):
        """执行count(*)查询的线程"""
        logger.info(f"Count query worker {worker_id} started")
        
        while self.running:
            try:
                with self.lock:
                    result = self.client.query(
                        collection_name=self.collection_name,
                        filter="",
                        output_fields=["count(*)"]
                    )
                
                count = result[0]["count(*)"]
                expected_count = self.total_records
                
                if count != expected_count:
                    logger.error(f"Worker {worker_id}: Count mismatch! Expected: {expected_count}, Got: {count}")
                    self.wrong_count += 1
                else:
                    logger.info(f"Worker {worker_id}: Count verified ✓ ({count})")
                
                time.sleep(random.uniform(0.5, 2.0))
                
            except Exception as e:
                logger.error(f"Error in count query worker {worker_id}: {e}")
                time.sleep(1)
    
    def run_concurrent_test(self, num_count_workers=5, test_duration=300):
        """运行并发测试"""
        logger.info(f"Starting concurrent test with {num_count_workers} count workers")
        logger.info(f"Test duration: {test_duration} seconds")
        
        # 启动upsert线程
        upsert_thread = threading.Thread(target=self.upsert_worker, daemon=True)
        upsert_thread.start()
        
        # 启动count查询线程
        count_threads = []
        for i in range(num_count_workers):
            thread = threading.Thread(
                target=self.count_query_worker, 
                args=(i,), 
                daemon=True
            )
            thread.start()
            count_threads.append(thread)
        
        # 运行指定时间
        try:
            time.sleep(test_duration)
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        
        # 停止测试
        self.running = False
        logger.info("Stopping concurrent test...")
        
        # 等待线程结束
        upsert_thread.join(timeout=5)
        for thread in count_threads:
            thread.join(timeout=5)
        
        # 最终验证
        self.final_verification()
        logger.info(f"Wrong count: {self.wrong_count}")
    
    def final_verification(self):
        """最终验证数据一致性"""
        logger.info("Performing final verification...")
        
        try:
            result = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["count(*)"]
            )
            
            final_count = result[0]["count(*)"]
            logger.info(f"Final count: {final_count}")
            
            if final_count == self.total_records:
                logger.info("✓ Final verification passed: count matches expected value")
            else:
                logger.error(f"✗ Final verification failed: expected {self.total_records}, got {final_count}")
                
        except Exception as e:
            logger.error(f"Error in final verification: {e}")

def main():
    """主函数"""
    test = ConcurrentTest()
    
    # 创建集合
    test.create_collection()
    
    # 插入初始数据
    test.insert_initial_data()
    
    # 运行并发测试
    test.run_concurrent_test(
        num_count_workers=10,    # 5个count查询线程
        test_duration=600       # 运行5分钟
    )

if __name__ == "__main__":
    main()
