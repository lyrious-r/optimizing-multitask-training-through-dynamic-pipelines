#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能日志系统使用示例
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynapipe.schedule_opt.execution_planner import get_performance_logger

def example_manual_logging():
    """示例：手动记录日志"""
    print("=== 手动记录日志示例 ===")
    
    # 设置日志目录
    log_dir = "./example_logs"
    os.environ["DYNAPIPE_PERFORMANCE_LOG_DIR"] = log_dir
    
    # 获取性能日志器
    perf_logger = get_performance_logger(log_dir)
    
    # 模拟一些函数执行
    functions = [
        ("process_batch", 0.5),
        ("optimize_schedule", 2.1),
        ("adjust_rc_plan", 0.3),
        ("generate_execution_plan", 3.2)
    ]
    
    for func_name, exec_time in functions:
        print(f"记录 {func_name} 执行时间: {exec_time} 秒")
        extra_params = {
            "batch_size": 4,
            "memory_limit": 8192,
            "extra_param": "example_value"
        }
        
        # 为adjust_rc_plan添加调用次数信息
        if func_name == "adjust_rc_plan":
            extra_params["call_count"] = 3  # 模拟这是第3次调用
            extra_params["topk"] = 1
            extra_params["over_limit_executors_count"] = 2
        
        # 为optimize_schedule添加adjust_rc_plan调用次数
        if func_name == "optimize_schedule":
            extra_params["adjust_rc_plan_call_count"] = 5  # 模拟总共调用了5次
        
        perf_logger.log_execution_time(
            func_name,
            exec_time,
            **extra_params
        )
    
    # 模拟一些RC计划
    rc_plans = [
        {
            "batch_idx": 1,
            "rc_plan": [[(0, 1, 0), (1, 2, 1), (2, 3, 0)]],
            "cost": 0.5,
            "method": "wait-free-cyclic"
        },
        {
            "batch_idx": 2,
            "rc_plan": [[(0, 1, 1), (1, 2, 0), (2, 3, 1)]],
            "cost": 0.7,
            "method": "cyclic"
        }
    ]
    
    for plan in rc_plans:
        print(f"记录 batch {plan['batch_idx']} 的RC计划")
        perf_logger.log_rc_plan(
            plan["batch_idx"],
            plan["rc_plan"],
            best_cost=plan["cost"],
            best_schedule_method=plan["method"],
            batch_size=2
        )
    
    print(f"\n日志已保存到: {log_dir}")
    print("可以运行以下命令查看分析结果:")
    print(f"python analyze_performance_logs.py --log-dir {log_dir}")

def example_automatic_logging():
    """示例：自动记录日志（需要实际的DynaPipe环境）"""
    print("\n=== 自动记录日志示例 ===")
    print("注意：这个示例需要完整的DynaPipe环境才能运行")
    print("在实际使用中，以下函数会自动记录执行时间：")
    print("- generate_execution_plan()")
    print("- optimize_schedule()")
    print("- adjust_rc_plan()")
    
    # 这里可以添加实际的DynaPipe调用示例
    # 但由于需要完整的模型和配置，这里只提供说明
    
    print("\n使用步骤：")
    print("1. 设置环境变量: export DYNAPIPE_PERFORMANCE_LOG_DIR='./my_logs'")
    print("2. 正常调用DynaPipe函数")
    print("3. 系统会自动记录执行时间和RC计划")
    print("4. 使用分析工具查看结果")

def main():
    print("DynaPipe 性能日志系统使用示例")
    print("=" * 50)
    
    # 运行手动记录示例
    example_manual_logging()
    
    # 运行自动记录示例说明
    example_automatic_logging()
    
    print("\n" + "=" * 50)
    print("示例完成！")
    print("\n相关文件：")
    print("- PERFORMANCE_LOGGING_README.md: 详细使用说明")
    print("- analyze_performance_logs.py: 日志分析工具")
    print("- test_performance_logger.py: 测试脚本")

if __name__ == "__main__":
    main() 