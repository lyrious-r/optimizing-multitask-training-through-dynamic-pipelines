#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试adjust_rc_plan调用次数记录功能
"""

import os
import sys
import time
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynapipe.schedule_opt.execution_planner import get_performance_logger

def test_adjust_rc_plan_counting():
    """测试adjust_rc_plan调用次数记录功能"""
    print("开始测试adjust_rc_plan调用次数记录功能...")
    
    # 设置日志目录
    log_dir = "./test_adjust_rc_plan_logs"
    os.environ["DYNAPIPE_PERFORMANCE_LOG_DIR"] = log_dir
    
    # 获取性能日志器
    perf_logger = get_performance_logger(log_dir)
    
    # 模拟多次optimize_schedule调用，每次有不同的adjust_rc_plan调用次数
    test_cases = [
        {
            "optimize_time": 1.5,
            "adjust_calls": 3,
            "sch_type": "wait-free-cyclic"
        },
        {
            "optimize_time": 2.1,
            "adjust_calls": 0,  # 没有调用adjust_rc_plan
            "sch_type": "cyclic"
        },
        {
            "optimize_time": 1.8,
            "adjust_calls": 5,
            "sch_type": "wait-free-cyclic"
        },
        {
            "optimize_time": 2.5,
            "adjust_calls": 2,
            "sch_type": "cyclic"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n模拟第{i}次optimize_schedule调用:")
        print(f"  执行时间: {case['optimize_time']} 秒")
        print(f"  adjust_rc_plan调用次数: {case['adjust_calls']}")
        print(f"  调度类型: {case['sch_type']}")
        
        # 记录optimize_schedule
        perf_logger.log_execution_time(
            "optimize_schedule",
            case["optimize_time"],
            sch_type=case["sch_type"],
            n_microbatches=4,
            memory_limit=8192,
            permutations_count=24,
            adjust_rc_plan_call_count=case["adjust_calls"]
        )
        
        # 模拟adjust_rc_plan的多次调用
        for call_num in range(1, case["adjust_calls"] + 1):
            adjust_time = 0.1 + (call_num * 0.05)  # 模拟递增的执行时间
            print(f"    记录adjust_rc_plan第{call_num}次调用: {adjust_time:.3f} 秒")
            
            perf_logger.log_execution_time(
                "adjust_rc_plan",
                adjust_time,
                memory_limit=8192,
                topk=1,
                over_limit_executors_count=2,
                call_count=call_num
            )
    
    # 检查日志文件
    performance_log_file = os.path.join(log_dir, "performance.log")
    
    print(f"\n检查日志文件: {performance_log_file}")
    
    if os.path.exists(performance_log_file):
        print("✓ 性能日志文件创建成功")
        
        # 解析日志并验证
        with open(performance_log_file, 'r') as f:
            logs = []
            for line in f:
                if "EXECUTION_TIME:" in line:
                    json_str = line.split("EXECUTION_TIME: ")[1].strip()
                    log_entry = json.loads(json_str)
                    logs.append(log_entry)
        
        # 分析结果
        optimize_logs = [log for log in logs if log["function"] == "optimize_schedule"]
        adjust_logs = [log for log in logs if log["function"] == "adjust_rc_plan"]
        
        print(f"\n📊 分析结果:")
        print(f"  optimize_schedule记录数: {len(optimize_logs)}")
        print(f"  adjust_rc_plan记录数: {len(adjust_logs)}")
        
        # 验证adjust_rc_plan调用次数
        total_adjust_calls = sum(log.get("adjust_rc_plan_call_count", 0) for log in optimize_logs)
        actual_adjust_logs = len(adjust_logs)
        
        print(f"  优化调度中记录的adjust_rc_plan总调用次数: {total_adjust_calls}")
        print(f"  实际记录的adjust_rc_plan日志数: {actual_adjust_logs}")
        
        if total_adjust_calls == actual_adjust_logs:
            print("✓ adjust_rc_plan调用次数记录正确")
        else:
            print("✗ adjust_rc_plan调用次数记录不匹配")
        
        # 显示详细的调用次数分布
        print(f"\n📋 调用次数分布:")
        for log in optimize_logs:
            call_count = log.get("adjust_rc_plan_call_count", 0)
            sch_type = log.get("sch_type", "unknown")
            print(f"  {sch_type}: {call_count} 次调用")
        
    else:
        print("✗ 性能日志文件创建失败")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_adjust_rc_plan_counting() 