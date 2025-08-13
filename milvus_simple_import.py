#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
    utility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import JSONL into Milvus with int PK, JSON field, and random vector field."
    )

    # Connection
    parser.add_argument(
        "--uri",
        default="http://localhost:19530",
        help="Milvus URI (default: http://localhost:19530)",
    )
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--token", default=None)

    # Collection
    parser.add_argument(
        "--collection",
        default="jsonl_collection",
        help="Collection name (default: jsonl_collection)",
    )
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate if exists")

    # Fields
    parser.add_argument("--pk-field", default="id", help="INT64 primary key field name (default: id)")
    parser.add_argument("--json-field", default="json", help="JSON field name (default: json)")
    parser.add_argument("--vector-field", default="vec", help="Vector field name (default: vec)")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension (default: 768)")

    # Input and batching
    parser.add_argument(
        "--file",
        default="data.jsonl",
        help="Path to JSONL file (default: data.jsonl)",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--pk-start", type=int, default=1, help="Starting PK value (default: 1)")
    parser.add_argument(
        "--rand-dist",
        choices=["uniform", "normal"],
        default="uniform",
        help="Random distribution for vectors (default: uniform)",
    )
    parser.add_argument("--progress-every", type=int, default=10000)

    # Index
    parser.add_argument("--create-index", action="store_true")
    parser.add_argument("--index-type", default="IVF_FLAT")
    parser.add_argument("--metric-type", default="L2")
    parser.add_argument("--index-params", default='{"nlist": 1024}')

    return parser.parse_args()


def connect(uri: str, user: Optional[str], password: Optional[str], token: Optional[str]) -> None:
    connections.disconnect(alias="default")
    connections.connect(alias="default", uri=uri, user=user, password=password, token=token)


def ensure_collection(
    name: str,
    pk_field: str,
    json_field: str,
    vector_field: str,
    dim: int,
    recreate: bool,
) -> Collection:
    if utility.has_collection(name):
        if recreate:
            print(f"Collection {name!r} exists; dropping as --recreate is set...")
            utility.drop_collection(name)
        else:
            print(f"Using existing collection {name!r}")
            return Collection(name)

    fields = [
        FieldSchema(name=pk_field, dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name=json_field, dtype=DataType.JSON),
        FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields=fields, description=f"Simple JSONL import: {name}")
    print(
        f"Creating collection {name!r} with fields: {pk_field}(INT64 PK), {json_field}(JSON), {vector_field}(FLOAT_VECTOR dim={dim})"
    )
    return Collection(name=name, schema=schema)


def parse_index_params(params_str: str) -> Dict[str, Any]:
    try:
        params = json.loads(params_str)
        if not isinstance(params, dict):
            raise ValueError
        return params
    except Exception:
        raise ValueError("--index-params must be a JSON object string, e.g. '{\"nlist\": 1024}'")


def create_index_if_needed(col: Collection, vector_field: str, index_type: str, metric_type: str, index_params_str: str) -> None:
    has_index = any(idx["field"] == vector_field for idx in col.indexes)
    if has_index:
        print(f"Index already exists on field {vector_field!r}; skipping index creation")
        return
    params = parse_index_params(index_params_str)
    print(f"Creating index on {vector_field!r}: index_type={index_type}, metric_type={metric_type}, params={params}")
    col.create_index(
        field_name=vector_field,
        index_params={
            "index_type": index_type,
            "metric_type": metric_type,
            "params": params,
        },
    )


def random_vector(dim: int, dist: str) -> List[float]:
    if dist == "normal":
        return [random.gauss(0.0, 1.0) for _ in range(dim)]
    return [random.random() for _ in range(dim)]


def iter_jsonl_rows(
    path: str,
    pk_field: str,
    json_field: str,
    vector_field: str,
    dim: int,
    pk_start: int,
    rand_dist: str,
) -> Iterable[Dict[str, Any]]:
    pk = pk_start
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("JSON line is not an object")
            except Exception as e:
                print(f"[WARN] Skip line {line_num}: {e}")
                continue

            row: Dict[str, Any] = {
                pk_field: pk,
                json_field: obj,
                vector_field: random_vector(dim, rand_dist),
            }
            pk += 1
            yield row


def insert_in_batches(
    col: Collection,
    rows: Iterable[Dict[str, Any]],
    batch_size: int,
    max_rows: Optional[int],
    progress_every: int,
) -> int:
    batch: List[Dict[str, Any]] = []
    total = 0
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            col.insert(batch)
            total += len(batch)
            batch = []
            if total % progress_every == 0:
                print(f"Inserted {total} rows...")
            if max_rows is not None and total >= max_rows:
                return total
    if batch:
        col.insert(batch)
        total += len(batch)
    return total


def load_with_index_fallback(
    col: Collection,
    vector_field: str,
    index_type: str,
    metric_type: str,
    index_params_str: str,
) -> None:
    try:
        col.load()
        return
    except Exception as e:
        if "index not found" in str(e).lower():
            print("[INFO] Load failed due to missing index; creating index and retrying load...")
            create_index_if_needed(col, vector_field, index_type, metric_type, index_params_str)
            col.load()
            return
        raise


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.file):
        print(f"Input file not found: {args.file}")
        sys.exit(1)

    connect(args.uri, args.user, args.password, args.token)
    col = ensure_collection(
        name=args.collection,
        pk_field=args.pk_field,
        json_field=args.json_field,
        vector_field=args.vector_field,
        dim=args.dim,
        recreate=args.recreate,
    )

    if args.create_index:
        create_index_if_needed(col, args.vector_field, args.index_type, args.metric_type, args.index_params)

    print(
        f"Starting import from {args.file!r} into collection {args.collection!r} (batch_size={args.batch_size})..."
    )
    inserted = insert_in_batches(
        col,
        rows=iter_jsonl_rows(
            path=args.file,
            pk_field=args.pk_field,
            json_field=args.json_field,
            vector_field=args.vector_field,
            dim=args.dim,
            pk_start=args.pk_start,
            rand_dist=args.rand_dist,
        ),
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
    )
    print(f"Inserted total rows: {inserted}")

    print("Flushing...")
    col.flush()

    if args.create_index:
        create_index_if_needed(col, args.vector_field, args.index_type, args.metric_type, args.index_params)

    print("Loading collection...")
    load_with_index_fallback(col, args.vector_field, args.index_type, args.metric_type, args.index_params)
    print("Done.")


if __name__ == "__main__":
    main()


