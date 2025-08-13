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
    enable_dynamic_field=True,
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="string_field", datatype=DataType.VARCHAR, max_length=100)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="id",
    index_type="STL_SORT"
)

index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 1024}
)

client.drop_collection(collection_name= "test_collection")
client.create_collection(
    collection_name="test_collection",
    schema=schema,
    index_params=index_params
)

colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "brown", "grey"]
data = []

for i in range(10):
    current_color = random.choice(colors)
    current_tag = random.randint(1000, 9999)
    current_coord = [ random.randint(0, 40) for _ in range(3) ]
    current_ref = [ [ random.choice(colors) for _ in range(3) ] for _ in range(3) ]
    data.append({
        "id": random.randint(0, 1000000),
        "string_field": "string_field_value",
        "vector": [ random.uniform(-1, 1) for _ in range(5) ],
            "label": current_color,
            "tag": current_tag,
            "coord": current_coord,
            "ref": current_ref
    })

for i in range(1):
    res = client.insert(
        collection_name="test_collection",
        data=data
    )

res = client.get_load_state(
    collection_name="test_collection"
)
print(res)

query_vectors = [ [ random.uniform(-1, 1) for _ in range(5) ]]
res = client.query(collection_name="test_collection", filter="id > 10", 
                   output_fields=["id", "$meta"], limit = 100)
print(res)
expr = "string_field == test_xxx"
# filter_params = {
#     "value": "test_xxx"
# }
print(expr)
res = client.search(
    collection_name="test_collection",
    data=query_vectors,
    # highlight-next-line
    filter=expr,
    #filter_params=filter_params,
    search_params={
        "metric_type": "L2",
        "params": {"nprobe": 16}
    },
    output_fields=["id", "color"],
    limit=3
)

print(res)


