      
import sys
import time
import functools

from datetime import datetime
import random

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

import logging

sample = 30

if len(sys.argv) >= 2:
    sample = int(sys.argv[1])

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT,
                    datefmt=DATE_FORMAT, handlers=handlers)
logger = logging.getLogger('LOGGER_NAME')

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 10000, 8

logger.info("start connecting to Milvus")
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("hello_milvus")
logger.info(f"Does collection hello_milvus exist in Milvus: {has}")
if has:
    utility.drop_collection("hello_milvus")
    logger.info("drop collection: hello_milvus")

rng = np.random.default_rng(seed=19530)

def my_sleep(interval):
    if False:
        time.sleep(interval)


def time_recorder(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start = datetime.now()
        res = func(*args, **kwargs)
        end = datetime.now()
        td = (end - start).total_seconds() * 10**3
        logger.info(
            f"The time of execution of {func.__name__} is : {td:.03f} ms")
        return res

    return inner


fields = [
    FieldSchema(name="pk", dtype=DataType.INT64,
                is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),

    FieldSchema(name="int64_array", dtype=DataType.ARRAY, max_capacity=128,
                element_type=DataType.INT64),
    FieldSchema(name="bool_array", dtype=DataType.ARRAY, max_capacity=128,
                element_type=DataType.BOOL),
    FieldSchema(name="float_array", dtype=DataType.ARRAY, max_capacity=128,
                element_type=DataType.FLOAT),
    FieldSchema(name="varchar_array", dtype=DataType.ARRAY, max_capacity=128,
                element_type=DataType.VARCHAR, max_length=1000),
]

schema = CollectionSchema(
    fields, "hello_milvus is the simplest demo to introduce the APIs")

logger.info("Create collection `hello_milvus`")
hello_milvus = Collection("hello_milvus", schema,
                          consistency_level="Eventually")


int64_samples = []
varchar_samples = []


def generate_random(total, sample):
    x = random.randint(0, total - 1) % sample
    return x


all_int64s = [generate_random(1000000, sample) for _ in range(5)]
all_varchars = [str(generate_random(1000000, sample)) for _ in range(5)]
logger.info(f"all_int64s: {all_int64s}")
logger.info(f"all_varchars: {all_varchars}")


@time_recorder
def generate_entities(offset_begin, num_entities, sample, total):
    logger.info(f"generate {num_entities} entities")
    int64_array_s = []
    bool_array_s = []
    float_array_s = []
    varchar_array_s = []

    for idx in range(num_entities):
        n = random.randint(1, 128)
        int64s = [generate_random(total, sample) for _ in range(n)]
        bools = [False]
        floats = [float(generate_random(total, sample)) for _ in range(n)]
        if (offset_begin + idx) % 100000 == 0:
            bools = [False, True]
        elif (offset_begin + idx) % 50000 == 0:
            int64s = all_int64s
            bools = [True]
            varchars = all_varchars
        varchars = [str(generate_random(total, sample)) for _ in range(n)]
        int64_array_s.append(int64s)
        bool_array_s.append(bools)
        float_array_s.append(floats)
        varchar_array_s.append(varchars)

    entities = [
        [i + offset_begin for i in range(num_entities)],
        rng.random((num_entities, dim)),
        int64_array_s,
        bool_array_s,
        float_array_s,
        varchar_array_s,
    ]
    return entities


@time_recorder
def insert():
    logger.info("Start inserting entities")
    n_batch = 5
    total = n_batch * num_entities
    for i in range(n_batch):
        entities = generate_entities(
            i * num_entities, num_entities, sample, total)
        hello_milvus.insert(entities)


@time_recorder
def flush():
    logger.info("start flushing")
    hello_milvus.flush()
    # check the num_entites
    logger.info(f"Number of entities in Milvus: {hello_milvus.num_entities}")


@time_recorder
def create_vector_index():
    logger.info("Start Creating index IVF_FLAT")
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    hello_milvus.create_index("embeddings", index)


@time_recorder
def create_scalar_index(index_type, field_name):
    logger.info(f"create_scalar_index {index_type} for {field_name}")
    #hello_milvus.create_index(field_name, {"index_type": index_type})
    hello_milvus.create_index(field_name)


@time_recorder
def load():
    logger.info("Start loading")
    hello_milvus.load()


@time_recorder
def release():
    logger.info("Start releasing")
    hello_milvus.release()


@time_recorder
def drop():
    logger.info("Start dropping")
    hello_milvus.drop()


@time_recorder
def run_query(expr, n, output_fields=["pk"]):
    logger.info(f"run {expr} {n} times with output fields: {output_fields}")
    for _ in range(n):
        hello_milvus.query(expr=expr, output_fields=output_fields)


def run_contains_query(n, expr_as_output=False):
    expr1 = f"array_contains(int64_array, 0)"
    expr2 = f"array_contains(bool_array, true)"
    expr3 = f"array_contains(varchar_array, '0')"
    if expr_as_output:
        run_query(expr1, n, ["int64_array"])
        run_query(expr2, n, ["bool_array"])
        run_query(expr3, n, ["varchar_array"])
    else:
        run_query(expr1, n)
        run_query(expr2, n)
        run_query(expr3, n)


def run_contains_any_query(n, expr_as_output=False):
    expr1 = f"array_contains_any(int64_array, {all_int64s})"
    expr2 = f"array_contains_any(bool_array, [true])"
    expr3 = f"array_contains_any(varchar_array, {all_varchars})"
    if expr_as_output:
        run_query(expr1, n, ["int64_array"])
        run_query(expr2, n, ["bool_array"])
        # varchar_array is too large.
        run_query(expr3, n, ["bool_array"])
    else:
        run_query(expr1, n)
        run_query(expr2, n)
        run_query(expr3, n)


def run_contains_all_query(n, expr_as_output=False):
    expr1 = f"array_contains_all(int64_array, {all_int64s})"
    expr2 = f"array_contains_all(bool_array, [true, false])"
    expr3 = f"array_contains_all(varchar_array, {all_varchars})"
    if expr_as_output:
        run_query(expr1, n, ["int64_array"])
        run_query(expr2, n, ["bool_array"])
        run_query(expr3, n, ["varchar_array"])
    else:
        run_query(expr1, n)
        run_query(expr2, n)
        run_query(expr3, n)


def run_array_equal_query(n, expr_as_output=False):
    expr1 = f"int64_array == {all_int64s}"
    expr2 = f"bool_array == [true, false]"
    expr3 = f"varchar_array == {all_varchars}"
    if expr_as_output:
        run_query(expr1, n, ["int64_array"])
        run_query(expr2, n, ["bool_array"])
        run_query(expr3, n, ["varchar_array"])
    else:
        run_query(expr1, n)
        run_query(expr2, n)
        run_query(expr3, n)


def run_all(n, field_name):
    logger.warn("run_all will hit every entity")
    expr = f'{field_name} >= 0'
    run_query(expr, n)


@time_recorder
def warm_query():
    logger.info(f"warm query, n: {n}, sample: {sample}")
    hello_milvus.query(expr="", output_fields=["count(*)"])
    hello_milvus.query(expr="pk % 100000 == 0", output_fields=["*"])


@time_recorder
def drop_index():
    logger.info("drop index")
    indexes = hello_milvus.indexes
    for index in indexes:
        logger.info(
            f"index info, field_name: {index.field_name}, index_name: {index.index_name}, index_params: {index.params}")
        if index.field_name != "embeddings":
            hello_milvus.drop_index(index_name=index.index_name)


n = 10

insert()
flush()
create_vector_index()

# create_scalar_index(index_type="STL_SORT", field_name=f"int64_array")
# create_scalar_index(index_type="STL_SORT", field_name=f"bool_array")
# create_scalar_index(index_type="STL_SORT", field_name=f"varchar_array")
# load()
# my_sleep(120)
# 
# warm_query()
# 
# my_sleep(120)
# 
# release()
# drop_index()

my_sleep(120)

create_scalar_index(index_type="BITMAP", field_name=f"int64_array")
create_scalar_index(index_type="BITMAP", field_name=f"bool_array")
#create_scalar_index(index_type="", field_name=f"float_array")
create_scalar_index(index_type="BITMAP", field_name=f"varchar_array")
load()
my_sleep(120)

warm_query()
#run_contains_query(n, False)
#run_contains_any_query(n, False)
#run_contains_all_query(n, False)
run_array_equal_query(n, False)
run_contains_query(n, True)
run_contains_any_query(n, True)
run_contains_all_query(n, True)
run_array_equal_query(n, True)

my_sleep(120)

release()
drop_index()

my_sleep(120)

load()
my_sleep(120)

warm_query()
run_contains_query(n, False)
run_contains_any_query(n, False)
run_contains_all_query(n, False)
run_array_equal_query(n, False)
run_contains_query(n, True)
run_contains_any_query(n, True)
run_contains_all_query(n, True)
run_array_equal_query(n, True)

my_sleep(120)

release()

my_sleep(120)

drop()

my_sleep(120)

    
