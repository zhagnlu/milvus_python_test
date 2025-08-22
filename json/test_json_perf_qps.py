from pymilvus import MilvusClient
import numpy as np
from loguru import logger
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

def ensure_loaded(client: MilvusClient, collection_name: str):
    # Load if API available; otherwise skip (some client versions auto-load on query)
    try:
        client.load_collection(collection_name=collection_name)
    except AttributeError:
        logger.warning("client.load_collection not available; skipping explicit load")


def percentile(arr_ms, p):
    if not arr_ms:
        return float("nan")
    return float(np.percentile(np.array(arr_ms, dtype=np.float64), p))


def run_benchmark(client: MilvusClient,
                  collection_name: str,
                  query_filter: str,
                  duration_seconds: float,
                  concurrency: int,
                  use_stats: bool = False,
                  print_result: bool = False):
    end_time = time.perf_counter() + duration_seconds
    latencies_ms_all = []
    latencies_lock = threading.Lock()
    success_count = 0
    error_count = 0
    counter_lock = threading.Lock()

    def worker():
        nonlocal success_count, error_count
        local_latencies = []
        filter_params = {"expr_use_json_stats": use_stats}
        while time.perf_counter() < end_time:
            t0 = time.perf_counter()
            try:
                res = client.query(
                    collection_name=collection_name,
                    filter=query_filter,
                    output_fields=["count(*)"],
                    filter_params=filter_params
                )
                if print_result:
                    logger.info(f"query res: {res}")
                t1 = time.perf_counter()
                local_latencies.append((t1 - t0) * 1000.0)
                with counter_lock:
                    success_count += 1
            except Exception as e:
                # Keep going; count errors
                with counter_lock:
                    error_count += 1
        # Merge local latencies at the end to reduce contention
        if local_latencies:
            with latencies_lock:
                latencies_ms_all.extend(local_latencies)

    start_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(concurrency):
            executor.submit(worker)
    # Threads auto-join at context exit
    end_wall = time.perf_counter()

    wall_seconds = max(end_wall - start_wall, 1e-9)
    total_requests = success_count + error_count
    qps = total_requests / wall_seconds

    p0 = min(latencies_ms_all) if latencies_ms_all else float("nan")
    p99 = percentile(latencies_ms_all, 99.0)
    avg_ms = (sum(latencies_ms_all) / len(latencies_ms_all)) if latencies_ms_all else float("nan")

    return {
        "qps": qps,
        "requests": total_requests,
        "success": success_count,
        "errors": error_count,
        "latency_ms_p0": p0,
        "latency_ms_p99": p99,
        "latency_ms_avg": avg_ms,
        "duration_s": wall_seconds,
        "concurrency": concurrency,
    }


def main():
    parser = argparse.ArgumentParser(description="JSON query QPS/latency benchmark (p0/p99)")
    parser.add_argument("--collection", default="json_perf_qps", help="collection name")
    parser.add_argument("--uri", default="http://localhost:19530", help="Milvus uri, e.g., http://localhost:19530")
    parser.add_argument("--token", default="", help="Milvus token if authentication is enabled")
    parser.add_argument("--list", action="store_true", help="list collections and exit")
    parser.add_argument("--duration", type=float, default=10.0, help="benchmark duration seconds")
    parser.add_argument("--concurrency", type=int, default=8, help="number of concurrent query workers")
    parser.add_argument("--filter",  help="filter expression to benchmark")
    parser.add_argument(
        "--use-stats",
        nargs='?',
        const='true',
        default='false',
        choices=['true', 'false'],
        help="set expr_use_json_stats (true/false). Bare --use-stats equals true. Default: false",
    )
    parser.add_argument(
        "--print-res",
        nargs='?',
        const='true',
        default='false',
        choices=['true', 'false'],
        help="print each query result (true/false). Bare equals true. Default: false",
    )
    args = parser.parse_args()
    if args.filter is None:
        logger.error("filter is required")
        return
    
    logger.info(f"connecting to {args.uri} ...")
    if args.token:
        client = MilvusClient(uri=args.uri, token=args.token)
    else:
        client = MilvusClient(uri=args.uri)
    logger.info("connected")

    # Optional: list collections and exit
    if args.list:
        try:
            cols = client.list_collections()
            logger.info(f"collections: {cols}")
        except Exception as e:
            logger.error(f"failed to list collections: {e}")
        return

    logger.info(f"loading collection: {args.collection}")
    ensure_loaded(client, args.collection)
    logger.info("collection loaded")

    # Validate collection exists before benchmark
    try:
        collections = client.list_collections()
        if args.collection not in collections:
            logger.error(f"collection not found: {args.collection}. Use --list to view available collections.")
            return
    except Exception as e:
        logger.error(f"failed to verify collection existence: {e}")
        return

    # Warm-up one query
    try:
        warmup_res = client.query(collection_name=args.collection, filter=args.filter, output_fields=["count(*)"])
        if str(args.print_res).lower() == 'true':
            logger.info(f"warm-up res: {warmup_res}")
    except Exception as e:
        logger.error(f"Warm-up query failed: {e}")
        return

    use_stats_bool = str(args.use_stats).lower() == 'true'
    print_res_bool = str(args.print_res).lower() == 'true'
    logger.info(f"running benchmark: duration={args.duration}s, concurrency={args.concurrency}, use_stats={use_stats_bool}, print_res={print_res_bool}")
    result = run_benchmark(
        client=client,
        collection_name=args.collection,
        query_filter=args.filter,
        duration_seconds=args.duration,
        concurrency=args.concurrency,
        use_stats=use_stats_bool,
        print_result=print_res_bool,
    )

    logger.info("Benchmark result:")
    msg = (
        "QPS={qps:.2f} req/s | Requests={requests} (success={success}, errors={errors}) | "
        "avg={avg:.2f} ms | p0={p0:.2f} ms | p99={p99:.2f} ms | duration={duration:.2f}s | conc={conc}"
    ).format(
        qps=result["qps"],
        requests=result["requests"],
        success=result["success"],
        errors=result["errors"],
        avg=result["latency_ms_avg"],
        p0=result["latency_ms_p0"],
        p99=result["latency_ms_p99"],
        duration=result["duration_s"],
        conc=result["concurrency"],
    )
    logger.info(msg)


if __name__ == "__main__":
    main()


