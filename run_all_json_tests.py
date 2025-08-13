#!/usr/bin/env python3
"""
JSON测试运行器
运行所有JSON相关的测试文件
"""

import sys
import time
import argparse
from loguru import logger
import subprocess
import os

def run_test_file(test_file, args=None):
    """运行单个测试文件"""
    logger.info(f"Running test file: {test_file}")
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return False
    
    try:
        start_time = time.time()
        
        # 构建命令
        cmd = [sys.executable, test_file]
        if args:
            cmd.extend(args)
        
        # 运行测试
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.success(f"✅ {test_file} completed successfully in {duration:.2f}s")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"❌ {test_file} failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {test_file} timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"💥 {test_file} failed with exception: {e}")
        return False

def run_basic_tests():
    """运行基础JSON测试"""
    logger.info("=== Running Basic JSON Tests ===")
    
    basic_tests = [
        "test-json.py",
        "test-json1.py", 
        "test_json.py",
        "test_json_perf.py"
    ]
    
    success_count = 0
    total_count = len(basic_tests)
    
    for test_file in basic_tests:
        if run_test_file(test_file):
            success_count += 1
    
    logger.info(f"Basic tests completed: {success_count}/{total_count} passed")
    return success_count == total_count

def run_comprehensive_tests():
    """运行综合JSON测试"""
    logger.info("=== Running Comprehensive JSON Tests ===")
    
    comprehensive_tests = [
        "test_json_comprehensive.py"
    ]
    
    success_count = 0
    total_count = len(comprehensive_tests)
    
    for test_file in comprehensive_tests:
        if run_test_file(test_file):
            success_count += 1
    
    logger.info(f"Comprehensive tests completed: {success_count}/{total_count} passed")
    return success_count == total_count

def run_edge_case_tests():
    """运行边界情况测试"""
    logger.info("=== Running Edge Case Tests ===")
    
    edge_case_tests = [
        "test_json_edge_cases.py"
    ]
    
    success_count = 0
    total_count = len(edge_case_tests)
    
    for test_file in edge_case_tests:
        if run_test_file(test_file):
            success_count += 1
    
    logger.info(f"Edge case tests completed: {success_count}/{total_count} passed")
    return success_count == total_count

def run_aggregation_tests():
    """运行聚合测试"""
    logger.info("=== Running Aggregation Tests ===")
    
    aggregation_tests = [
        "test_json_aggregation.py"
    ]
    
    success_count = 0
    total_count = len(aggregation_tests)
    
    for test_file in aggregation_tests:
        if run_test_file(test_file):
            success_count += 1
    
    logger.info(f"Aggregation tests completed: {success_count}/{total_count} passed")
    return success_count == total_count

def run_real_world_tests():
    """运行真实世界场景测试"""
    logger.info("=== Running Real-world Scenario Tests ===")
    
    real_world_tests = [
        "test_json_real_world.py"
    ]
    
    success_count = 0
    total_count = len(real_world_tests)
    
    for test_file in real_world_tests:
        if run_test_file(test_file):
            success_count += 1
    
    logger.info(f"Real-world tests completed: {success_count}/{total_count} passed")
    return success_count == total_count

def run_json_folder_tests():
    """运行json文件夹中的测试"""
    logger.info("=== Running JSON Folder Tests ===")
    
    json_folder_tests = [
        "json/test-json.py",
        "json/random_json.py"
    ]
    
    success_count = 0
    total_count = len(json_folder_tests)
    
    for test_file in json_folder_tests:
        if run_test_file(test_file):
            success_count += 1
    
    logger.info(f"JSON folder tests completed: {success_count}/{total_count} passed")
    return success_count == total_count

def run_specific_test(test_name):
    """运行特定的测试"""
    logger.info(f"=== Running Specific Test: {test_name} ===")
    
    # 检查测试文件是否存在
    test_files = [
        test_name,
        f"{test_name}.py",
        f"test_{test_name}.py",
        f"test-{test_name}.py"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            return run_test_file(test_file)
    
    logger.error(f"Test file not found: {test_name}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Run JSON tests for Milvus")
    parser.add_argument("--test-type", choices=[
        "basic", "comprehensive", "edge-cases", "aggregation", 
        "real-world", "json-folder", "all"
    ], default="all", help="Type of tests to run")
    
    parser.add_argument("--specific-test", help="Run a specific test file")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    logger.info("🚀 Starting JSON Test Runner")
    logger.info(f"Python version: {sys.version}")
    
    start_time = time.time()
    
    if args.specific_test:
        success = run_specific_test(args.specific_test)
    else:
        if args.test_type == "all":
            # 运行所有测试
            results = []
            results.append(run_basic_tests())
            results.append(run_comprehensive_tests())
            results.append(run_edge_case_tests())
            results.append(run_aggregation_tests())
            results.append(run_real_world_tests())
            results.append(run_json_folder_tests())
            success = all(results)
        elif args.test_type == "basic":
            success = run_basic_tests()
        elif args.test_type == "comprehensive":
            success = run_comprehensive_tests()
        elif args.test_type == "edge-cases":
            success = run_edge_case_tests()
        elif args.test_type == "aggregation":
            success = run_aggregation_tests()
        elif args.test_type == "real-world":
            success = run_real_world_tests()
        elif args.test_type == "json-folder":
            success = run_json_folder_tests()
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info("=" * 50)
    if success:
        logger.success(f"🎉 All tests completed successfully in {total_duration:.2f}s")
        return 0
    else:
        logger.error(f"💥 Some tests failed. Total duration: {total_duration:.2f}s")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 