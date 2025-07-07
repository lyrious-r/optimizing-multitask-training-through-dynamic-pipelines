#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能日志分析工具
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Any

def parse_performance_log(log_file: str) -> List[Dict[str, Any]]:
    """解析性能日志文件"""
    logs = []
    if not os.path.exists(log_file):
        print(f"警告: 日志文件 {log_file} 不存在")
        return logs
    
    with open(log_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            if "EXECUTION_TIME:" in line:
                try:
                    # 提取JSON部分
                    json_str = line.split("EXECUTION_TIME: ")[1].strip()
                    log_entry = json.loads(json_str)
                    logs.append(log_entry)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                except Exception as e:
                    print(f"警告: 第{line_num}行处理失败: {e}")
    
    return logs

def parse_rc_plans(rc_plan_file: str) -> List[Dict[str, Any]]:
    """解析RC计划文件"""
    if not os.path.exists(rc_plan_file):
        print(f"警告: RC计划文件 {rc_plan_file} 不存在")
        return []
    
    try:
        with open(rc_plan_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: RC计划文件JSON解析失败: {e}")
        return []
    except Exception as e:
        print(f"错误: 读取RC计划文件失败: {e}")
        return []

def analyze_execution_times(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析执行时间"""
    if not logs:
        return {}
    
    # 按函数分组
    function_stats = defaultdict(list)
    adjust_rc_plan_stats = defaultdict(list)  # 专门统计adjust_rc_plan的调用次数
    
    for log in logs:
        function_name = log.get("function", "unknown")
        execution_time = log.get("execution_time", 0)
        function_stats[function_name].append(execution_time)
        
        # 如果是adjust_rc_plan，记录调用次数
        if function_name == "adjust_rc_plan":
            call_count = log.get("call_count", 0)
            adjust_rc_plan_stats["call_counts"].append(call_count)
    
    # 计算统计信息
    analysis = {}
    for func_name, times in function_stats.items():
        if times:
            analysis[func_name] = {
                "count": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "times": sorted(times)
            }
    
    # 添加adjust_rc_plan调用次数统计
    if adjust_rc_plan_stats["call_counts"]:
        analysis["adjust_rc_plan_call_counts"] = {
            "total_calls": sum(adjust_rc_plan_stats["call_counts"]),
            "avg_calls_per_optimize": sum(adjust_rc_plan_stats["call_counts"]) / len(adjust_rc_plan_stats["call_counts"]),
            "min_calls": min(adjust_rc_plan_stats["call_counts"]),
            "max_calls": max(adjust_rc_plan_stats["call_counts"]),
            "call_counts": sorted(adjust_rc_plan_stats["call_counts"])
        }
    
    return analysis

def analyze_rc_plans(rc_plans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析RC计划"""
    if not rc_plans:
        return {}
    
    analysis = {
        "total_batches": len(rc_plans),
        "batch_indices": [plan.get("batch_idx", 0) for plan in rc_plans],
        "costs": [plan.get("best_cost", 0) for plan in rc_plans],
        "schedule_methods": [plan.get("best_schedule_method", "unknown") for plan in rc_plans],
        "batch_sizes": [plan.get("batch_size", 0) for plan in rc_plans]
    }
    
    # 统计调度方法使用情况
    method_counts = defaultdict(int)
    for method in analysis["schedule_methods"]:
        method_counts[method] += 1
    analysis["schedule_method_distribution"] = dict(method_counts)
    
    # 计算成本统计
    if analysis["costs"]:
        analysis["cost_stats"] = {
            "min_cost": min(analysis["costs"]),
            "max_cost": max(analysis["costs"]),
            "avg_cost": sum(analysis["costs"]) / len(analysis["costs"])
        }
    
    return analysis

def print_analysis(execution_analysis: Dict[str, Any], rc_analysis: Dict[str, Any]):
    """打印分析结果"""
    print("=" * 60)
    print("性能日志分析报告")
    print("=" * 60)
    
    # 执行时间分析
    if execution_analysis:
        print("\n📊 执行时间分析:")
        print("-" * 40)
        for func_name, stats in execution_analysis.items():
            if func_name == "adjust_rc_plan_call_counts":
                continue  # 跳过，单独处理
            print(f"\n函数: {func_name}")
            print(f"  调用次数: {stats['count']}")
            print(f"  总执行时间: {stats['total_time']:.4f} 秒")
            print(f"  平均执行时间: {stats['avg_time']:.4f} 秒")
            print(f"  最短执行时间: {stats['min_time']:.4f} 秒")
            print(f"  最长执行时间: {stats['max_time']:.4f} 秒")
        
        # 显示adjust_rc_plan调用次数统计
        if "adjust_rc_plan_call_counts" in execution_analysis:
            call_stats = execution_analysis["adjust_rc_plan_call_counts"]
            print(f"\n🔄 adjust_rc_plan 调用次数统计:")
            print(f"  总调用次数: {call_stats['total_calls']}")
            print(f"  每次optimize_schedule平均调用: {call_stats['avg_calls_per_optimize']:.2f} 次")
            print(f"  最少调用次数: {call_stats['min_calls']}")
            print(f"  最多调用次数: {call_stats['max_calls']}")
            print(f"  调用次数分布: {call_stats['call_counts']}")
    else:
        print("\n❌ 没有找到执行时间日志")
    
    # RC计划分析
    if rc_analysis:
        print("\n📋 RC计划分析:")
        print("-" * 40)
        print(f"总批次数: {rc_analysis['total_batches']}")
        
        if "cost_stats" in rc_analysis:
            cost_stats = rc_analysis["cost_stats"]
            print(f"成本统计:")
            print(f"  最低成本: {cost_stats['min_cost']:.4f}")
            print(f"  最高成本: {cost_stats['max_cost']:.4f}")
            print(f"  平均成本: {cost_stats['avg_cost']:.4f}")
        
        if "schedule_method_distribution" in rc_analysis:
            print(f"调度方法分布:")
            for method, count in rc_analysis["schedule_method_distribution"].items():
                percentage = (count / rc_analysis["total_batches"]) * 100
                print(f"  {method}: {count} 次 ({percentage:.1f}%)")
    else:
        print("\n❌ 没有找到RC计划日志")
    
    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description="分析性能日志")
    parser.add_argument("--log-dir", default="./performance_logs", 
                       help="日志目录路径")
    parser.add_argument("--performance-log", 
                       help="性能日志文件路径 (默认: log_dir/performance.log)")
    parser.add_argument("--rc-plan-log", 
                       help="RC计划文件路径 (默认: log_dir/rc_plans.json)")
    
    args = parser.parse_args()
    
    # 确定文件路径
    if args.performance_log:
        performance_log_file = args.performance_log
    else:
        performance_log_file = os.path.join(args.log_dir, "performance.log")
    
    if args.rc_plan_log:
        rc_plan_file = args.rc_plan_log
    else:
        rc_plan_file = os.path.join(args.log_dir, "rc_plans.json")
    
    print(f"分析日志目录: {args.log_dir}")
    print(f"性能日志文件: {performance_log_file}")
    print(f"RC计划文件: {rc_plan_file}")
    
    # 解析日志
    print("\n正在解析日志文件...")
    performance_logs = parse_performance_log(performance_log_file)
    rc_plans = parse_rc_plans(rc_plan_file)
    
    print(f"找到 {len(performance_logs)} 条执行时间记录")
    print(f"找到 {len(rc_plans)} 条RC计划记录")
    
    # 分析数据
    execution_analysis = analyze_execution_times(performance_logs)
    rc_analysis = analyze_rc_plans(rc_plans)
    
    # 打印分析结果
    print_analysis(execution_analysis, rc_analysis)

if __name__ == "__main__":
    main() 