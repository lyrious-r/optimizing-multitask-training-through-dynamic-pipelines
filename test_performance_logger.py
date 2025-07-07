#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试性能日志系统的脚本
"""

import os
import sys
import time
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynapipe.schedule_opt.execution_planner import get_performance_logger

def test_performance_logger():
    """测试性能日志系统"""
    print("开始测试性能日志系统...")
    
    # 设置日志目录
    log_dir = "./test_performance_logs"
    os.environ["DYNAPIPE_PERFORMANCE_LOG_DIR"] = log_dir
    
    # 获取性能日志器
    perf_logger = get_performance_logger(log_dir)
    
    # 测试执行时间日志
    print("测试执行时间日志...")
    perf_logger.log_execution_time(
        "test_function", 
        1.234,
        param1="value1",
        param2=42
    )
    
    # 测试rc_plan日志
    print("测试rc_plan日志...")
    test_rc_plan = [
        [(0, 1, 0), (1, 2, 1), (2, 3, 0)],
        [(0, 1, 1), (1, 2, 0), (2, 3, 1)]
    ]
    perf_logger.log_rc_plan(
        1,
        test_rc_plan,
        batch_size=2,
        cost=0.5
    )
    
    # 检查日志文件是否创建
    performance_log_file = os.path.join(log_dir, "performance.log")
    rc_plan_file = os.path.join(log_dir, "rc_plans.json")
    
    print(f"检查日志文件...")
    print(f"性能日志文件: {performance_log_file}")
    print(f"RC计划文件: {rc_plan_file}")
    
    if os.path.exists(performance_log_file):
        print("✓ 性能日志文件创建成功")
        with open(performance_log_file, 'r') as f:
            content = f.read()
            print(f"性能日志内容:\n{content}")
    else:
        print("✗ 性能日志文件创建失败")
    
    if os.path.exists(rc_plan_file):
        print("✓ RC计划文件创建成功")
        with open(rc_plan_file, 'r') as f:
            content = json.load(f)
            print(f"RC计划内容:\n{json.dumps(content, indent=2)}")
    else:
        print("✗ RC计划文件创建失败")
    
    print("测试完成!")

if __name__ == "__main__":
    test_performance_logger() 