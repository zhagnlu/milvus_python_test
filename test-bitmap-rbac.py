import random, time
from pymilvus import connections, MilvusClient, DataType

CLUSTER_ENDPOINT = "http://localhost:19530"

# 1. Set up a Milvus client
client = MilvusClient(
    uri=CLUSTER_ENDPOINT
)

# 2. Create a collection
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=False,
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="data", datatype=DataType.VARCHAR, max_length=100)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="security_group", datatype=DataType.ARRAY, 
                 element_type=DataType.VARCHAR, max_capacity=10, max_length=100)

index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 1024}
)

index_params.add_index(field_name="security_group", 
                       index_type="BITMAP")

client.create_collection(
    collection_name="test_collection",
    schema=schema,
    index_params=index_params
)
data =[]
data.append({
        "id": random.randint(0, 100000),
        "vector": [ random.uniform(-1, 1) for _ in range(128) ],
        "data": "data" + str(random.randint(0,100000)),
        "security_group": ["ceo"]
})

data.append({
        "id": random.randint(0, 100000),
        "vector": [ random.uniform(-1, 1) for _ in range(128) ],
        "data": "data" + str(random.randint(0,100000)),
        "security_group": ["finance"]
})

data.append({
        "id": random.randint(0, 100000),
        "vector": [ random.uniform(-1, 1) for _ in range(128) ],
        "data": "data" + str(random.randint(0,100000)),
        "security_group": ["sales"]
})

data.append({
        "id": random.randint(0, 100000),
        "vector": [ random.uniform(-1, 1) for _ in range(128) ],
        "data": "data" + str(random.randint(0,100000)),
        "security_group": ["develop"]
})

data.append({
        "id": random.randint(0, 100000),
        "vector": [ random.uniform(-1, 1) for _ in range(128) ],
        "data": "data" + str(random.randint(0,100000)),
        "security_group": ["sales", "develop"]
})

res = client.insert(
        collection_name="test_collection",
        data=data)


query_vectors = [ [ random.uniform(-1, 1) for _ in range(128) ]]
res = client.query(
    collection_name="test_collection",
    # 查询仅 ceo role 可见的数据
    filter='array_contains(security_group, "ceo")',
    output_fields=["id", "data", "security_group"],
)
print("ceo role read:")
print(res)

res = client.query(
    collection_name="test_collection",
    # 查询仅 sales role 可见的数据
    filter='array_contains(security_group, "sales")',
    output_fields=["id", "data", "security_group"],
)
print("sales role read:")
print(res)

res = client.query(
    collection_name="test_collection",
    # 查询仅 develop 可见的数据
    filter='array_contains(security_group, "develop")',
    output_fields=["id", "data", "security_group"],
)
print("develop role read:")
print(res)

res = client.query(
    collection_name="test_collection",
    # 查询仅 develop 或者 ceo 可见的数据
    filter='array_contains_any(security_group, ["develop", "ceo"])',
    output_fields=["id", "data", "security_group"],
)
print("develop or ceo role read:")
print(res)

upsert_vector =  [ random.uniform(-1, 1) for _ in range(128) ]
upsert_data = "data" + str(random.randint(0,100000))
upsert_row_raw = {
        "id": 101,
        "vector": upsert_vector,
        "data": upsert_data,
        "security_group": ["finance"]
}
data.append(upsert_row_raw)

res = client.insert(
    collection_name="test_collection",
    data=data)
res = client.query(
    collection_name="test_collection",
    filter='id==101',
    output_fields=["*"],
)
print("pk = 101:")
print(res)

upsert_row_update = {
        "id": 101,
        "vector": upsert_vector,
        "data": upsert_data,
        "security_group": ["finance", "sales"]
}
res = client.upsert(
    collection_name="test_collection",
    data=upsert_row_update)
print("after upsert")
res = client.query(
    collection_name="test_collection",
    filter='id==101',
    output_fields=["*"],
)
print(res)


