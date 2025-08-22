
from pymilvus import MilvusClient, DataType, connections, utility, Collection
import numpy as np
import random
from loguru import logger
import time
import random_json
import pprint
import json
import argparse
import glob
import os

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


def insert_collection_streaming(collection_name, file_path, batch_size, pk_start, max_rows=None, progress_every=10000):
    """Insert rows from a JSONL file. Returns (inserted_total, json_len_total, json_count)."""
    inserted_total = 0
    json_len_total = 0
    json_count = 0
    next_id = pk_start
    batch = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if max_rows is not None and inserted_total >= max_rows:
                break
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    logger.warning(f"Skip line {line_num}: JSON is not an object")
                    continue
            except Exception as e:
                logger.warning(f"Skip line {line_num}: {e}")
                continue

            row = {
                "my_id": next_id,
                "my_vector": [random.random() for _ in range(dim)],
                "json": obj,
            }
            batch.append(row)
            next_id += 1
            json_len_total += len(s)
            json_count += 1

            if len(batch) >= batch_size:
                client.insert(collection_name=collection_name, data=batch)
                inserted_total += len(batch)
                batch = []
                if inserted_total % progress_every == 0:
                    logger.info(f"Inserted {inserted_total} rows...")

        # tail
        if batch:
            client.insert(collection_name=collection_name, data=batch)
            inserted_total += len(batch)

    return inserted_total, json_len_total, json_count


def insert_multiple_files(collection_name, file_pattern, batch_size, pk_start, max_rows=None, progress_every=10000):
    files = glob.glob(file_pattern)
    if not files:
        logger.error(f"No files found matching pattern: {file_pattern}")
        return 0
    
    files.sort()  # 确保文件顺序一致
    logger.info(f"Found {len(files)} files: {files}")
    
    inserted_total = 0
    json_len_total = 0
    json_count = 0
    next_id = pk_start
    
    for file_path in files:
        if not os.path.isfile(file_path):
            logger.warning(f"Skipping non-file: {file_path}")
            continue
            
        logger.info(f"Processing file: {file_path}")
        file_inserted, file_len_total, file_count = insert_collection_streaming(
            collection_name=collection_name,
            file_path=file_path,
            batch_size=batch_size,
            pk_start=next_id,
            max_rows=max_rows,
            progress_every=progress_every,
        )
        inserted_total += file_inserted
        json_len_total += file_len_total
        json_count += file_count
        next_id += file_inserted
        
        if max_rows is not None and inserted_total >= max_rows:
            logger.info(f"Reached max rows limit: {max_rows}")
            break
    
    return inserted_total, json_len_total, json_count


def connect_milvus(uri=None, user=None, password=None, token=None):
    """连接到指定的Milvus实例"""
    try:
        if uri:
            # 断开现有连接
            connections.disconnect(alias="default")
            # 建立新连接
            connections.connect(alias="default", uri=uri, user=user, password=password, token=token)
            logger.info(f"Connected to Milvus: {uri}")
        else:
            logger.info("Using default Milvus connection")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise


def query_collection(collection_name, query_expr=None, output_fields=None, limit=10, offset=0):
    """查询集合数据"""
    try:
        if not utility.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist")
            return None
        
        collection = Collection(collection_name)
        collection.load()
        
        # 默认查询所有字段
        if output_fields is None:
            output_fields = ["my_id", "my_vector", "json"]
        
        # 构建查询表达式
        if query_expr is None:
            query_expr = "my_id >= 0"  # 默认查询所有
        
        logger.info(f"Querying collection {collection_name} with expr: {query_expr}")
        logger.info(f"Output fields: {output_fields}")
        logger.info(f"Limit: {limit}, Offset: {offset}")
        
        results = collection.query(
            expr=query_expr,
            output_fields=output_fields,
            limit=limit,
            offset=offset
        )
        
        logger.info(f"Query returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return None


def list_collections():
    """列出所有集合"""
    try:
        collections = utility.list_collections()
        logger.info(f"Found {len(collections)} collections: {collections}")
        return collections
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []


def get_collection_info(collection_name):
    """获取集合信息"""
    try:
        if not utility.has_collection(collection_name):
            logger.error(f"Collection {collection_name} does not exist")
            return None
        
        collection = Collection(collection_name)
        info = {
            "name": collection_name,
            "schema": collection.schema,
            "num_entities": collection.num_entities,
            "indexes": collection.indexes,
            "is_empty": collection.is_empty
        }
        
        logger.info(f"Collection {collection_name} info:")
        logger.info(f"  Schema: {info['schema']}")
        logger.info(f"  Num entities: {info['num_entities']}")
        logger.info(f"  Indexes: {info['indexes']}")
        logger.info(f"  Is empty: {info['is_empty']}")
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Import JSONL: int PK (increment), random vector, JSON field from file')
    
    # 连接参数
    parser.add_argument('--uri', help='Milvus URI (e.g., http://localhost:19530)')
    parser.add_argument('--user', help='Milvus username')
    parser.add_argument('--password', help='Milvus password')
    parser.add_argument('--token', help='Milvus token')
    
    # 操作模式
    parser.add_argument('--mode', choices=['import', 'query', 'list', 'info'], default='import',
                       help='Operation mode: import, query, list collections, or get collection info')
    
    # 导入相关参数
    parser.add_argument('--file', help='Path to single JSONL file (one JSON per line)')
    parser.add_argument('--files', help='Glob pattern for multiple files (e.g., "data/*.jsonl" or "file_*.jsonl")')
    parser.add_argument('--collection', default="jsonl_collection", help='Collection name')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch insert size (default: 1000)')
    parser.add_argument('--pk-start', type=int, default=0, help='Starting PK value (default: 0)')
    parser.add_argument('--max-rows', type=int, default=None, help='Max rows to import (default: all)')
    parser.add_argument('--progress-every', type=int, default=10000, help='Print progress every N rows')
    parser.add_argument('--create', action='store_true', help='Drop and recreate collection before insert')
    
    # 查询相关参数
    parser.add_argument('--query-expr', help='Query expression (e.g., "my_id > 100")')
    parser.add_argument('--output-fields', nargs='+', help='Output fields for query')
    parser.add_argument('--limit', type=int, default=10, help='Query result limit (default: 10)')
    parser.add_argument('--offset', type=int, default=0, help='Query result offset (default: 0)')
    
    args = parser.parse_args()

    # 连接Milvus
    if args.uri:
        connect_milvus(args.uri, args.user, args.password, args.token)

    if args.mode == 'import':
        # 导入模式
        if args.create:
            create_collection_base(args.collection)

        if args.file and args.files:
            logger.error("Cannot specify both --file and --files")
            return
        
        if not args.file and not args.files:
            logger.error("Must specify either --file or --files")
            return

        if args.file:
            # 单文件模式
            inserted, json_len_total, json_count = insert_collection_streaming(
                collection_name=args.collection,
                file_path=args.file,
                batch_size=args.batch_size,
                pk_start=args.pk_start,
                max_rows=args.max_rows,
                progress_every=args.progress_every,
            )
        else:
            # 多文件模式
            inserted, json_len_total, json_count = insert_multiple_files(
                collection_name=args.collection,
                file_pattern=args.files,
                batch_size=args.batch_size,
                pk_start=args.pk_start,
                max_rows=args.max_rows,
                progress_every=args.progress_every,
            )

        logger.info(f"Inserted total rows: {inserted}")
        if json_count > 0:
            avg_json_len = json_len_total / float(json_count)
            logger.info(f"Average JSON length (chars) of inserted rows: {avg_json_len:.2f}")
        else:
            logger.info("No JSON rows inserted; average JSON length unavailable")

        client.flush(collection_name=args.collection)
        client.load_collection(collection_name=args.collection)
        
    elif args.mode == 'query':
        # 查询模式
        results = query_collection(
            collection_name=args.collection,
            query_expr=args.query_expr,
            output_fields=args.output_fields,
            limit=args.limit,
            offset=args.offset
        )
        if results:
            print("\nQuery Results:")
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                for field, value in result.items():
                    if field == 'my_vector':
                        print(f"{field}: [{value[0]:.4f}, {value[1]:.4f}, ..., {value[-1]:.4f}] (dim={len(value)})")
                    elif field == 'json':
                        print(f"{field}: {str(value)[:100]}...")
                    else:
                        print(f"{field}: {value}")
                        
    elif args.mode == 'list':
        # 列出所有集合
        collections = list_collections()
        
    elif args.mode == 'info':
        # 获取集合信息
        info = get_collection_info(args.collection)


if __name__ == '__main__':
    main()