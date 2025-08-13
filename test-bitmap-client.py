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
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
# highlight-next-line
schema.add_field(field_name="bool", datatype=DataType.BOOL)
schema.add_field(field_name="int8", datatype=DataType.INT8)
schema.add_field(field_name="int16", datatype=DataType.INT16)
schema.add_field(field_name="int32", datatype=DataType.INT32)
schema.add_field(field_name="int64", datatype=DataType.INT64)
schema.add_field(field_name="float", datatype=DataType.FLOAT)
schema.add_field(field_name="double", datatype=DataType.DOUBLE)
schema.add_field(field_name="string", datatype=DataType.VARCHAR)
schema.add_field(field_name="json", datatype=DataType.JSON)

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

client.create_collection(
    collection_name="test_collection",
    schema=schema,
    index_params=index_params
)

colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "brown", "grey"]
data = []

for i in range(1000):
    current_color = random.choice(colors)
    current_tag = random.randint(1000, 9999)
    current_coord = [ random.randint(0, 40) for _ in range(3) ]
    current_ref = [ [ random.choice(colors) for _ in range(3) ] for _ in range(3) ]
    data.append({
        "id": random.randint(0, 1000000),
        "vector": [ random.uniform(-1, 1) for _ in range(5) ],
        "color": {
            "label": current_color,
            "tag": current_tag,
            "coord": current_coord,
            "ref": current_ref
        }
    })

for i in range(1000):
    res = client.insert(
        collection_name="test_collection",
        data=data
    )

res = client.get_load_state(
    collection_name="test_collection"
)
print(res)

query_vectors = [ [ random.uniform(-1, 1) for _ in range(5) ]]
res = client.search(
    collection_name="test_collection",
    data=query_vectors,
    # highlight-next-line
    filter='color["label"] in ["red"]',
    search_params={
        "metric_type": "L2",
        "params": {"nprobe": 16}
    },
    output_fields=["id", "color"],
    limit=3
)

print(res)


