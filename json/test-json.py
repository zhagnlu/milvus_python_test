from pymilvus import MilvusClient, DataType
import numpy as np
import random
from loguru import logger
import time
import random_json
import pprint

client = MilvusClient()
logger.info("connected")


dim=128
logger.info("creating index params")
schema = client.create_schema(enable_dynamic_field=False)
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="json", datatype=DataType.JSON)
logger.info("created schema")
logger.info(schema)

use_stats_param = {"expr_use_json_stats": True}
not_use_stats_param = {"expr_use_json_stats": False}

json_template_data = [1, np.float64(1.0), np.double(1.0), 9707199254740993.0, 9707199254740992,
                     '1', '123', '321', '213', True, False, 123, 1.12, 100, 1.34325, False, 3, 'xxx', 'aaa', 
                     1, 34, 1.345, [1, 2], [1.0, 2], None, {}, {"a": 1},
                     {'a': 1.0}, {'a': 9707199254740993.0}, {'a': 9707199254740992}, {'a': '1'},  {'a': '123'}, {'a': '321'}, {'a': '213'}, {'a': True},
                     {'a': [1, 2, 3]}, {'a': [1.0, 2, '1']}, {'a': [1.0, 2]}, {'a': None}, {'a': {'b': 1}}, {'a': {'b': 1.0}}, {'a': [{'b': 1}, 2.0, np.double(3.0), '4', True, [1, 3.0], None]}]

def create_collection_base(collection_name):
    client.drop_collection(collection_name)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info("created collection done")

def insert_collection_base(collection_name, nb_single, jsons):
    data = [{
            "my_id": j ,
            "my_vector": [random.random() for _ in range(dim)],
            "json": jsons[j],
    } for j in range(nb_single)]
        
    print("len(str(data[0]['json'])): ", len(str(data[0]['json'])))
    res = client.insert(collection_name=collection_name, data=data)
    logger.info(f"Inserted {nb_single} records")
    client.flush(collection_name=collection_name)
    client.load_collection(collection_name=collection_name)

def test0():
    collection_name = "json_unary_test0"   
    create_collection_base(collection_name)
    
    jsons = []
    for j in range(0, 6000):
        json_map = { }
        jsons.append(json_map)
    insert_collection_base(collection_name, 6000, jsons)
    client.load_collection(collection_name=collection_name)
    while 1:
        res=client.query(collection_name=collection_name, filter="json['key0'] == 1", output_fields=['count(*)'])
        print(res)
        time.sleep(1)
        res=client.query(collection_name=collection_name, filter="json['key0'] == 1", output_fields=['count(*)'])
        print(res)
        time.sleep(1)

def test1(init=False):
    # test null value
    collection_name = "json_unary_test1"   
    if init:
        create_collection_base(collection_name)
        jsons = []
        for j in range(0, 6000):
            if j % 3 == 1:
                json_map = { "key0": random.choice(json_template_data) }
            elif j % 3 == 2:
                json_map = { "key0": None }
            else:
                json_map = {}
            print(json_map)
            jsons.append(json_map)
        print("len(jsons): ", len(jsons))
        insert_collection_base(collection_name, 6000, jsons)
    client.load_collection(collection_name=collection_name)
    index = 0
    while 1:
        # test empty string
        print("expr: json['key0'] == ''")
        res=client.query(collection_name=collection_name, filter="json['key0'] == ''", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res=client.query(collection_name=collection_name, filter="json['key0'] == ''", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        # test null
        print("expr: json['key0'] is null")
        res=client.query(collection_name=collection_name, filter="json['key0'] is null", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res=client.query(collection_name=collection_name, filter="json['key0'] is null", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        # test json is null
        print("expr: json is null")
        res=client.query(collection_name=collection_name, filter="json is null", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res=client.query(collection_name=collection_name, filter="json is null", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)


def test2(init=False):
    # test single key with different type of value
    collection_name = "json_unary_test2"
    if init:
        create_collection_base(collection_name)
        jsons = []
        for j in range(0, 6000):
            json_map = { "key0": random.choice(json_template_data) }
            jsons.append(json_map)
        insert_collection_base(collection_name, 6000, jsons)
    client.load_collection(collection_name=collection_name)
    index = 0
    while 1:
        print("expr: json['key0'] == 1")
        res=client.query(collection_name=collection_name, filter="json['key0'] == 1", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res=client.query(collection_name=collection_name, filter="json['key0'] == 1", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0'] == '1'")
        res=client.query(collection_name=collection_name, filter="json['key0'] == '1'", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res=client.query(collection_name=collection_name, filter="json['key0'] == '1'", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0'] == 1.0")
        res=client.query(collection_name=collection_name, filter="json['key0'] == 1.0", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res=client.query(collection_name=collection_name, filter="json['key0'] == 1.0", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0/a'] == 1.0")
        res=client.query(collection_name=collection_name, filter="json['key0']['a'] == 1", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res=client.query(collection_name=collection_name, filter="json['key0']['a'] == 1", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        res = client.query(collection_name=collection_name, limit=10000, filter="", output_fields=['json'])
        for i, item in enumerate(res):
            print(f"[{i}] {item}")
        #pprint.pprint(res, width=10000)

def test3(init=False):
    # test all non-shared keys
    collection_name = "json_unary_test3"
    if init:
        create_collection_base(collection_name)
        jsons = []
        for j in range(0, 6000):
            if j % 10 > 3:
                json_map = { "key0": 10}
            else:
                json_map = { "key0": '100'}
            jsons.append(json_map)
        insert_collection_base(collection_name, 6000, jsons)
    client.load_collection(collection_name=collection_name)
    index = 0
    while 1:

        print("expr: json['key0'] == 10")
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0'] == '100'")
        res = client.query(collection_name=collection_name, filter="json['key0'] == '100'", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == '100'", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

def test4(init=False):
    # test all shared-and-non-shared keys but all same keys
    collection_name = "json_unary_test4"
    if init:
        create_collection_base(collection_name)
        jsons = []
        for j in range(0, 6000):
            if j % 10 > 1:
                json_map = { "key0": 10}
            else:
                json_map = { "key0": '100'}
            jsons.append(json_map)
        insert_collection_base(collection_name, 6000, jsons)
    client.load_collection(collection_name=collection_name)
    index = 0
    while 1:

        print("expr: json['key0'] == 10")
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0'] == '100'")
        res = client.query(collection_name=collection_name, filter="json['key0'] == '100'", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == '100'", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

def test5(init=False):
    # test all shared-and-non-shared keys but all same keys
    collection_name = "json_unary_test5"
    if init:
        create_collection_base(collection_name)
        jsons = []
        for j in range(0, 6000):
            if j % 10 > 3:
                json_map = { "key0": 10}
            elif j % 10 == 2:
                json_map = { "key0": 10.0}
            else:
                json_map = { "key0": '100'}
            jsons.append(json_map)
        insert_collection_base(collection_name, 6000, jsons)
    client.load_collection(collection_name=collection_name)
    index = 0
    while 1:

        print("expr: json['key0'] == 10")
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0'] == 10.0")
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10.0", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10.0", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0'] == 10.0")
        res = client.query(collection_name=collection_name, filter="json['key0'] > 9.6", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] > 9.6", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)
        
        print("expr: json['key0'] == '100'")
        res = client.query(collection_name=collection_name, filter="json['key0'] == '100'", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == '100'", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1) 


def test6(init=False):
    # test all shared-and-non-shared keys but all same keys
    collection_name = "json_unary_test6"
    if init:
        create_collection_base(collection_name)
        jsons = []
        for j in range(0, 6000):
            if j % 10 > 5:
                json_map = { "key0": 10}
            elif j % 10 in [2, 3, 4, 5]:
                json_map = { "key0": 10.0}
            else:
                json_map = { "key0": '100'}
            jsons.append(json_map)
        insert_collection_base(collection_name, 6000, jsons)
    client.load_collection(collection_name=collection_name)
    index = 0
    while 1:

        print("expr: json['key0'] == 10")
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0'] == 10.0")
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10.0", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == 10.0", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)

        print("expr: json['key0'] == 10.0")
        res = client.query(collection_name=collection_name, filter="json['key0'] > 9.6", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] > 9.6", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1)
        
        print("expr: json['key0'] == '100'")
        res = client.query(collection_name=collection_name, filter="json['key0'] == '100'", output_fields=['count(*)'], filter_params=use_stats_param)
        print(res)
        res = client.query(collection_name=collection_name, filter="json['key0'] == '100'", output_fields=['count(*)'], filter_params=not_use_stats_param)
        print(res)
        print("====")
        time.sleep(1) 
        
if __name__ == "__main__":
    #test1(True)
    #test2(True)
    #test3(True)
    test4(True)
    #test5(True)
    #test6(True)