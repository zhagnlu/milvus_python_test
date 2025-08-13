from pymilvus import MilvusClient, DataType
import numpy as np
import random
from loguru import logger
import time

client = MilvusClient()
logger.info("connected")

dim=128
logger.info("creating index params")
schema = client.create_schema(enable_dynamic_field=False)
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="int16_array", datatype=DataType.ARRAY, element_type=DataType.INT16, max_capacity=5)

logger.info("created schema")
logger.info(schema)


res = client.list_collections()

collection_name = "array_index_simple_try_1"
if collection_name in res:
   client.drop_collection(collection_name)

logger.info("preparing index params")
index_params = client.prepare_index_params()
index_params.add_index(field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE")
logger.info("prepared index params")
logger.info("creating collection")
client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
logger.info("created collection")


nb = 3000
vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
data = np.int16(2)

data = [{"my_id": i, "my_vector": vectors[i], "int16_array": [i, i + 1] } for i in range(nb)]

res = client.insert(collection_name=collection_name, data=data)
logger.info(res)
logger.info(f"inserted {nb}")

client.flush(collection_name)
logger.info("flushed")

total_num = client.query(collection_name=collection_name, filter="json_contains(int16_array, 1)", output_fields=["count(*)"])
logger.info("Total number for expr json_contains(int16_array, 1) before index:")
logger.info(total_num)

res = client.query(collection_name=collection_name, filter="json_contains(int16_array, 1)", output_fields=["int16_array"])
logger.info("The result for expr json_contains(int16_array, 1) before index:")
logger.info(res)

client.release_collection(collection_name)
logger.info("released collection")
client.drop_index(collection_name, "my_vector")
logger.info("dropped index")
logger.info("preparing another index params")
index_params = client.prepare_index_params()
index_params.add_index(field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index(field_name="int16_array", index_type="INVERTED")
logger.info("creating array index")
client.create_index(collection_name, index_params)
logger.info("created array index done")
client.load_collection(collection_name)
logger.info("loaded collection")


total_num = client.query(collection_name=collection_name, filter="json_contains(int16_array, 1)", output_fields=["count(*)"])
logger.info("Total number for expr json_contains(int16_array, 1) after index:")
logger.info(total_num)

res = client.query(collection_name=collection_name, filter="json_contains(int16_array, 1)", output_fields=["int16_array"])
logger.info("The result for expr json_contains(int16_array, 1) after index:")
logger.info(res)