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

#_URI = "https://in01-b73b98f8b163591.aws-us-west-2.vectordb.zillizcloud.com:19540"  #2.4
_URI = "https://in01-d53a5a04b7589b7.aws-us-west-2.vectordb.zillizcloud.com:19540" #2.5 

_TOKEN = "f4490f28d41a81d27af8dd7c48ff8d98e41e628351769edce57603f52ecef5aea2d9cfd61ea803626b25d30568ced185ef358e1f"
_DB_NAME = "default"
collection_name = "item_tower_embedding_1116"

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

# collection.release()
# collection.drop_index(index_name='sw_lat')
# #collection.drop_index(index_name='timestamp_index')
# print("drop index done")
# collection.create_index('sw_lat', {
#  'index_type': "STL_SORT"
# })
# print("create index done")
collection.load()
print("load done")

#result = collection.query(expr="timestamp > 1748139869 and (is_fulltime==true or is_contract==true) and (work_model=='REMOTE' or (work_model=='ON_SITE') or (work_model=='HYBRID')) and (is_exp_junior_v2==true)", output_fields=["count(*)"])
#print(result)

while(1):
    result = collection.query(expr="work_model=='REMOTE'", output_fields=["count(*)"])
    print(result)
# while(1):
#     result = collection.query(expr="(work_model=='REMOTE' or (work_model=='ON_SITE' and (ne_lng>=-74.930271 and sw_lng<=-73.021616 and ne_lat>=40.037259 and sw_lat<=41.486535)) or (work_model=='HYBRID' and (ne_lng>=-74.930271 and sw_lng<=-73.021616 and ne_lat>=40.037259 and sw_lat<=41.486535)))", output_fields=["count(*)"])
#     print(result)
# while (1):
#     result = collection.query(expr="(ne_lng>=-74.930271 and sw_lng<=-73.021616 and ne_lat>=40.037259 and sw_lat<=41.486535) ", output_fields=["count(*)"])
#     print(result)

# while (1):
#     result = collection.query(expr="(timestamp>1747568534) and (is_fulltime==true or is_contract==true) and ((work_model=='ON_SITE' and (ne_lng>=-77.202621 and sw_lng<=-75.398349 and ne_lat>=36.025806 and sw_lat<=37.475081)) or (work_model=='HYBRID' and (ne_lng>=-77.202621 and sw_lng<=-75.398349 and ne_lat>=36.025806 and sw_lat<=37.475081)) or (work_model=='ON_SITE' and (ne_lng>=-84.819481 and sw_lng<=-83.949656 and ne_lat>=33.434765 and sw_lat<=34.159402)) or \
#                          (work_model=='HYBRID' and (ne_lng>=-84.819481 and sw_lng<=-83.949656 and ne_lat>=33.434765 and sw_lat<=34.159402)) or (work_model=='ON_SITE' and (ne_lng>=-97.963934 and sw_lng<=-97.076375 and ne_lat>=35.108753 and sw_lat<=35.833391)) or (work_model=='HYBRID' and (ne_lng>=-97.963934 and sw_lng<=-97.076375 and ne_lat>=35.108753 and sw_lat<=35.833391)) or (work_model=='ON_SITE' and (ne_lng>=-77.924084 and sw_lng<=-77.012567 and ne_lat>=37.170571 and sw_lat<=37.895209)) \
#                          or (work_model=='HYBRID' and (ne_lng>=-77.924084 and sw_lng<=-77.012567 and ne_lat>=37.170571 and sw_lat<=37.895209)) or (work_model=='ON_SITE' and (ne_lng>=-79.097008 and sw_lng<=-78.205545 and ne_lat>=35.459310 and sw_lat<=36.183948)) or (work_model=='HYBRID' and (ne_lng>=-79.097008 and sw_lng<=-78.205545 and ne_lat>=35.459310 and sw_lat<=36.183948)) or (work_model=='ON_SITE' and (ne_lng>=-97.784109 and sw_lng<=-96.870633 and ne_lat>=37.330306 and sw_lat<=38.054944)) or \
#                         (work_model=='HYBRID' and (ne_lng>=-97.784109 and sw_lng<=-96.870633 and ne_lat>=37.330306 and sw_lat<=38.054944)) or (work_model=='ON_SITE' and (ne_lng>=-98.186655 and sw_lng<=-97.349424 and ne_lat>=29.941185 and sw_lat<=30.665822)) or (work_model=='HYBRID' and (ne_lng>=-98.186655 and sw_lng<=-97.349424 and ne_lat>=29.941185 and sw_lat<=30.665822))) and (is_exp_new_grad_v2==true or is_exp_junior_v2==true)", output_fields=["count(*)"])
#     print(result)






