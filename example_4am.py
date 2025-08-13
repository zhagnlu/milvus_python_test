from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

import time
import numpy

_URI = "https://in01-067c5c13729567f.aws-us-west-2.vectordb.zillizcloud.com:19539"
_TOKEN = "zcloud_root:O1;1U.o^:r-O7<v)q.Rjt3]tOLY,*RP["
_DB_NAME = "default"
collection_name = "vanchat_openai_3_large_3072"

print(f"connect to milvus\n")
connections.connect(
        uri=_URI,
        token=_TOKEN,
        db_name=_DB_NAME
)
print("connect done")

try:
    collections = utility.list_collections()
    print("Collections in Milvus:")
    for collection in collections:
        print(f"- {collection}")
except Exception as e:
    print(f"Error listing collections: {e}")
  
# Access the collection
try:
    collection = Collection(collection_name)
    print(f"Collection '{collection_name}' is ready for searching.")
except Exception as e:
    print(f"Error accessing collection '{collection_name}': {e}")
    exit()

result = collection.query(expr='pk == "7ddb4e15-7c16-42bb-96dc-98927da5b85c" ', output_fields=["*"])
print(result)

result = collection.query(expr='pk == "8085ddcb-2ed9-4673-9775-1612f4115e21" ', output_fields=["*"])
print(result)




