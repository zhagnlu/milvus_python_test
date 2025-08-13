from pymilvus import MilvusClient, DataType
import numpy as np
import random
from loguru import logger
import time
import random_json

client = MilvusClient()
logger.info("connected")


dim=128
logger.info("creating index params")
schema = client.create_schema(enable_dynamic_field=True)
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)
logger.info("created schema")
logger.info(schema)

res = client.list_collections()

collection_name = "json_path_index_dynamic_0"
if 1:
    if collection_name in res:
       client.drop_collection(collection_name)
    
    logger.info("preparing index params")
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE")
    logger.info("prepared index params")
    logger.info("creating collection")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info("created collection")
    res = client.describe_collection(collection_name)
    logger.info(res)
    index_list = client.list_indexes(collection_name=collection_name)
    logger.info(index_list)
    for index_name in index_list:
       res = client.describe_index(collection_name, index_name)
       logger.info(res)
    
    inserted_data = [1, np.float64(1.0), np.double(1.0), 9707199254740993.0, 9707199254740992,
                     '1', '123', '321', '213', True, False, 123, 1.12, 100, 1.34325, False, 3, 'xxx', 'aaa', 
                     1, 34, 1.345, [1, 2], [1.0, 2], None, {}, {"a": 1},
                     {'a': 1.0}, {'a': 9707199254740993.0}, {'a': 9707199254740992}, {'a': '1'},  {'a': '123'}, {'a': '321'}, {'a': '213'}, {'a': True},
                     {'a': [1, 2, 3]}, {'a': [1.0, 2, '1']}, {'a': [1.0, 2]}, {'a': None}, {'a': {'b': 1}}, {'a': {'b': 1.0}}, {'a': [{'b': 1}, 2.0, np.double(3.0), '4', True, [1, 3.0], None]}]
                   
    logger.info(len(inserted_data))
    #assert len(inserted_data) == 40
    nb_single = 6000
    circle = 0
    # Loop over each element in inserted_data
    value_set = ["int", "float", "bool", "str"]
    json_map = {}
    #json_map = { f"key{i}": random.choice(value_set) for i in range(0, 20)}
    while 1:
        #Generate data for each test case
        if circle % 10 < 1:
            json_map = { f"key{i}": random.choice(value_set) for i in range(0, 100) }
        else:
            json_map = {  f"key{i}": random.choice(value_set) for i in range(1, 100) }
        # json_map = {  f"key{i}": random.choice(value_set) for i in range(circle * 6000, 6000 *circle + 20) }
        data = [{
            "my_id": j + circle * nb_single,
            "my_vector": [random.random() for _ in range(dim)],
            "my_varchar": "varchar",
            "my_json": json_map,
            #"my_json": random_json.generate_random_json_with_fixed_schema(json_map)   # Use the test case here
        } for j in range(nb_single)]
        
        print(len(str(data[0]['my_json'])))
        res = client.insert(collection_name=collection_name, data=data)
        logger.info(f"Inserted {nb_single} records")
        circle += 1
        if circle > 30:
            break  # Safety break if needed
    
    client.flush(collection_name=collection_name)
    client.load_collection(collection_name=collection_name)


client.load_collection(collection_name=collection_name)
print("load done")

#res=client.query(collection_name=collection_name, filter="my_id == 11999", output_fields=['my_json'])
#print(res)
while 1:
  res=client.query(collection_name=collection_name, filter=" my_json['key0'] == 'int'", output_fields=['count(*)'])
  #res=client.query(collection_name=collection_name, filter=" varchar == 'int'", output_fields=['count(*)'])
  print(res)
# res=client.query(collection_name=collection_name, filter="my_json == 1", output_fields=['count(*)'])
# print(res)q
