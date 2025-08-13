import time
import numpy as np
from pymilvus import (
    MilvusClient,
    DataType
)

fmt = "\n=== {:30} ===\n"
dim = 128
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("content", DataType.VARCHAR, max_length=64)


index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "embeddings", metric_type="L2")
milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Bounded")

print(fmt.format("    all collections    "))
print(milvus_client.list_collections())

print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))

rng = np.random.default_rng(seed=19530)
rows = [
        {"id": 1, "embeddings": rng.random((1, dim))[0], "content": "akx"},
        {"id": 2, "embeddings": rng.random((1, dim))[0], "content": "bkx"},
      # {"id": 3, "embeddings": rng.random((1, dim))[0],"content": "ckx"},
        {"id": 4, "embeddings": rng.random((1, dim))[0], "content": "dkx"},
        {"id": 5, "embeddings": rng.random((1, dim))[0], "content": "ekx"},
        {"id": 6, "embeddings": rng.random((1, dim))[0], "content": "fkx"},
]

for i in range(0, 500):
    content = f"{i}kd"
    if i == 250:
        rows.append({"id": 3, "embeddings": rng.random((1, dim))[0], "content": "ckx"})
    rows.append({"id": 7 + i, "embeddings": rng.random((1,dim))[0], "content":content})

milvus_client.create_index(collection_name, index_params)

print(fmt.format("Start inserting entities"))
print("len of rows", len(rows))
insert_result = milvus_client.insert(collection_name, rows)
print(fmt.format("Inserting entities done"))

#milvus_client.flush(collection_name)
#time.sleep(100)

print(fmt.format("Start load collection "))
milvus_client.load_collection(collection_name)

print(fmt.format("Start query by specifying filtering expression"))
query_results = milvus_client.query(collection_name, filter= '((content like "%x")&&(content like "c%"))', limit=10, output_fields=['content'], consistency_level="Strong")
for ret in query_results: 
    print(ret)

print(fmt.format("Start query by count *"))
query_results = milvus_client.query(collection_name, filter= '((content like "%x"))', limit=10, output_fields=['content'], consistency_level="Strong")
print(query_results)

# print(fmt.format("Start query by specifying pk "))
# expr = 'id == 3'
# query_results = milvus_client.query(collection_name, filter= expr, limit=10, output_fields=['content'])

# for ret in query_results: 
#     print(ret)

#milvus_client.drop_collection(collection_name)
