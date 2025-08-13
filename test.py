from pymilvus import MilvusClient, DataType, MilvusException, AnnSearchRequest, RRFRanker, WeightedRanker
# connect to database
client = MilvusClient("http://127.0.0.1:19530", dbname="testdb")
# reset database
for c in client.list_collections():
    client.release_collection(c)
    client.drop_collection(c)
# create schema
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="f0", datatype=DataType.ARRAY, nullable=False, element_type=DataType.FLOAT, max_capacity=15)
schema.add_field(field_name="f1", datatype=DataType.FLOAT, nullable=True)
schema.add_field(field_name="f2", datatype=DataType.VARCHAR, is_primary=True, nullable=False, max_length=10)
schema.add_field(field_name="f3", datatype=DataType.VARCHAR, is_primary=False, nullable=False, max_length=10)
schema.add_field(field_name="f4", datatype=DataType.FLOAT_VECTOR, dim=36)
schema.add_field(field_name="f5", datatype=DataType.FLOAT_VECTOR, dim=7)
# create collection
client.create_collection(collection_name="test_collection", schema=schema)

data_list = [
{'f0': [0.13565], 'f1': 0.74628, 'f2': 'Bhhh', 'f3': '', 'f4': [0.10183, 0.41692, 0.09707, 0.80158, 0.71121, 0.42691, 0.21145, 0.70464, 0.7903, 0.59769, 0.41238, 0.19156, 0.39654, 0.45484, 0.64215, 0.68114, 0.57937, 0.92467, 0.33257, 0.03504, 0.92324, 0.45617, 0.30679, 0.68081, 0.44642, 0.82858, 0.54887, 0.52261, 0.2558, 0.56181, 0.07956, 0.27769, 0.08695, 0.78347, 0.77027, 0.2285], 'f5': [0.72764, 0.55666, 0.42759, 0.56282, 0.05871, 0.76042, 0.22816]},
]
client.upsert(collection_name='test_collection', data=data_list)
client.flush("test_collection")
# create index
index_params = client.prepare_index_params()
params = {}
index_params.add_index(field_name="f0", index_type="INVERTED", metric_type="", index_name="i0", params=params)
client.create_index(collection_name="test_collection", index_params=index_params)
# create index
index_params = client.prepare_index_params()
params = {'nlist': 56646}
index_params.add_index(field_name="f4", index_type="IVF_FLAT", metric_type="COSINE", index_name="i1", params=params)
client.create_index(collection_name="test_collection", index_params=index_params)
# create index
index_params = client.prepare_index_params()
params = {'nlist': 8913}
index_params.add_index(field_name="f5", index_type="IVF_FLAT", metric_type="IP", index_name="i2", params=params)
client.create_index(collection_name="test_collection", index_params=index_params)
client.load_collection(collection_name="test_collection")

data_list = [
{'f0': [0.32943999767303467, 0.8565000295639038, -0.06499999761581421, -0.7717900276184082, 0.0, 0.0, 0.5043900012969971, -0.8187800049781799, 0.0, -0.09502000361680984], 'f1': -0.3075200021266937, 'f3': '', 'f4': [0.07840999960899353, 0.3929600119590759, 0.3891200125217438, 0.8228499889373779, 0.4120500087738037, 0.1926099956035614, 0.2264299988746643, 0.38512998819351196, 0.2621400058269501, 0.9724599719047546, 0.7569299936294556, 0.20654000341892242, 0.35697999596595764, 0.44336000084877014, 0.556190013885498, 0.6926900148391724, 0.9784200191497803, 0.9030299782752991, 0.28988999128341675, 0.03536999970674515, 0.626579999923706, 0.2707799971103668, 0.2950800061225891, 0.6023200154304504, 0.06464999914169312, 0.5862299799919128, 0.6502500176429749, 0.5175999999046326, 0.24028000235557556, 0.60794997215271, 0.6734899878501892, 0.14172999560832977, 0.3631500005722046, 0.8116400241851807, 0.15738999843597412, 0.20534999668598175], 'f5': [-1.0, -0.08489, 0.84771, 0.55457, -0.65019, 1.0, 0.67231], 'f2': 'Xv2SOT'},
]
client.upsert(collection_name='test_collection', data=data_list)
client.flush("test_collection")
# create index
index_params = client.prepare_index_params()
params = {}
index_params.add_index(field_name="f0", index_type="INVERTED", metric_type="", index_name="i0", params=params)
client.create_index(collection_name="test_collection", index_params=index_params)
# create index
index_params = client.prepare_index_params()
params = {'nlist': 56646}
index_params.add_index(field_name="f4", index_type="IVF_FLAT", metric_type="COSINE", index_name="i1", params=params)
client.create_index(collection_name="test_collection", index_params=index_params)
# create index
index_params = client.prepare_index_params()
params = {'nlist': 8913}
index_params.add_index(field_name="f5", index_type="IVF_FLAT", metric_type="IP", index_name="i2", params=params)
client.create_index(collection_name="test_collection", index_params=index_params)

import sys

try:
    res = client.search(
                            collection_name='test_collection',
                            data=[[0.67358, 0, 0, 0, -0.61603, -0.72258, -0.05721]],
                            anns_field='f5', 
                            filter='ARRAY_CONTAINS_ANY(f0, ["xxx"])', 
                            limit=3848, 
                            output_fields=['f0', 'f1', 'f2', 'f3', 'f4', 'f5'], 
                            search_params={'params': {'nprobe': 8913}})
except MilvusException as e:
    if 'Assert "(value_proto.val_case() == milvus::proto::plan::GenericValue::kFloatVal)"' in e.message:
        print("trigger target bug!!")
        sys.exit(0)
    else:
        print("not the original error")
        sys.exit(-1)

print("normal case, no bug")
sys.exit(-1)