# 性能日志系统使用说明

## 概述

这个性能日志系统用于记录DynaPipe调度优化过程中的执行时间和rc_plan方案。系统会自动记录以下信息：

1. `generate_execution_plan` 函数的执行时间
2. `optimize_schedule` 函数的执行时间  
3. `adjust_rc_plan` 函数的执行时间
4. **每次`optimize_schedule`运行中`adjust_rc_plan`的执行次数**
5. 最终的rc_plan方案

## 功能特性

### 1. 执行时间记录
- 自动记录关键函数的执行时间
- 包含函数参数和上下文信息
- 支持自定义额外参数

### 2. RC计划记录
- 记录每个batch的最终rc_plan方案
- 包含batch索引、执行成本等信息
- 以JSON格式存储，便于后续分析

### 3. adjust_rc_plan调用次数统计
- 记录每次`optimize_schedule`运行中`adjust_rc_plan`被调用的次数
- 帮助分析内存优化策略的效果
- 统计平均调用次数、最大/最小调用次数等

### 4. 灵活的日志目录配置
- 支持通过环境变量配置日志目录
- 默认日志目录：`./performance_logs`
- 环境变量：`DYNAPIPE_PERFORMANCE_LOG_DIR`

## 使用方法

### 1. 基本使用

```python
from dynapipe.schedule_opt.execution_planner import get_performance_logger

# 获取性能日志器
perf_logger = get_performance_logger()

# 手动记录执行时间
perf_logger.log_execution_time(
    "my_function", 
    1.234,  # 执行时间（秒）
    param1="value1",
    param2=42
)

# 记录rc_plan
perf_logger.log_rc_plan(
    batch_idx=1,
    rc_plan=[[(0, 1, 0), (1, 2, 1)]],
    batch_size=2,
    cost=0.5
)
```

### 2. 自动记录

当调用以下函数时，系统会自动记录执行时间：

- `generate_execution_plan()` - 记录整个执行计划生成过程
- `optimize_schedule()` - 记录调度优化过程，包括adjust_rc_plan调用次数
- `adjust_rc_plan()` - 记录重计算计划调整过程

### 3. 配置日志目录

```bash
# 设置环境变量
export DYNAPIPE_PERFORMANCE_LOG_DIR="/path/to/your/logs"

# 或者在Python中设置
import os
os.environ["DYNAPIPE_PERFORMANCE_LOG_DIR"] = "/path/to/your/logs"
```

## 日志文件格式

### 1. 性能日志文件 (`performance.log`)

每行包含一个JSON格式的日志条目：

```json
{
  "function": "optimize_schedule",
  "execution_time": 1.234,
  "timestamp": 1640995200.0,
  "sch_type": "wait-free-cyclic",
  "n_microbatches": 4,
  "memory_limit": 8192,
  "permutations_count": 24,
  "adjust_rc_plan_call_count": 5
}
```

adjust_rc_plan的日志条目：
```json
{
  "function": "adjust_rc_plan",
  "execution_time": 0.123,
  "timestamp": 1640995200.0,
  "memory_limit": 8192,
  "topk": 1,
  "over_limit_executors_count": 2,
  "call_count": 3
}
```

### 2. RC计划文件 (`rc_plans.json`)

包含所有batch的rc_plan方案：

```json
[
  {
    "batch_idx": 1,
    "rc_plan": [
      [[0, 1, 0], [1, 2, 1], [2, 3, 0]],
      [[0, 1, 1], [1, 2, 0], [2, 3, 1]]
    ],
    "timestamp": 1640995200.0,
    "best_cost": 0.5,
    "best_schedule_method": "wait-free-cyclic",
    "batch_size": 2
  }
]
```

## 日志分析

### 1. 查看执行时间统计

```python
import json
import pandas as pd

# 读取性能日志
with open("performance_logs/performance.log", "r") as f:
    logs = []
    for line in f:
        if line.startswith("[") and "EXECUTION_TIME:" in line:
            # 提取JSON部分
            json_str = line.split("EXECUTION_TIME: ")[1]
            log_entry = json.loads(json_str)
            logs.append(log_entry)

# 转换为DataFrame进行分析
df = pd.DataFrame(logs)
print(df.groupby("function")["execution_time"].describe())
```

### 2. 分析RC计划

```python
# 读取RC计划
with open("performance_logs/rc_plans.json", "r") as f:
    rc_plans = json.load(f)

# 分析每个batch的RC计划
for plan in rc_plans:
    print(f"Batch {plan['batch_idx']}: cost={plan['best_cost']}")
    print(f"RC Plan: {plan['rc_plan']}")
```

### 3. 分析adjust_rc_plan调用次数

```python
# 分析adjust_rc_plan调用次数
adjust_calls = [log for log in logs if log["function"] == "adjust_rc_plan"]
if adjust_calls:
    call_counts = [log["call_count"] for log in adjust_calls]
    print(f"adjust_rc_plan调用次数统计:")
    print(f"  总调用次数: {sum(call_counts)}")
    print(f"  平均每次optimize_schedule调用: {sum(call_counts)/len(call_counts):.2f} 次")
    print(f"  最多调用次数: {max(call_counts)}")
    print(f"  最少调用次数: {min(call_counts)}")
```

## 分析工具使用

使用内置的分析工具查看完整统计：

```bash
python analyze_performance_logs.py --log-dir ./performance_logs
```

输出示例：
```
============================================================
性能日志分析报告
============================================================

📊 执行时间分析:
----------------------------------------

函数: optimize_schedule
  调用次数: 10
  总执行时间: 15.2340 秒
  平均执行时间: 1.5234 秒
  最短执行时间: 0.8234 秒
  最长执行时间: 2.1234 秒

函数: adjust_rc_plan
  调用次数: 25
  总执行时间: 2.3450 秒
  平均执行时间: 0.0938 秒
  最短执行时间: 0.0234 秒
  最长执行时间: 0.2340 秒

🔄 adjust_rc_plan 调用次数统计:
  总调用次数: 25
  每次optimize_schedule平均调用: 2.50 次
  最少调用次数: 0
  最多调用次数: 5
  调用次数分布: [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]

📋 RC计划分析:
----------------------------------------
总批次数: 10
成本统计:
  最低成本: 0.4500
  最高成本: 0.8900
  平均成本: 0.6234
调度方法分布:
  wait-free-cyclic: 8 次 (80.0%)
  cyclic: 2 次 (20.0%)

============================================================
```

## 测试

运行测试脚本验证系统功能：

```bash
python test_performance_logger.py
```

## 注意事项

1. **日志目录权限**：确保程序有权限在指定目录创建文件
2. **磁盘空间**：长时间运行可能产生大量日志，注意磁盘空间
3. **性能影响**：日志记录会带来轻微的性能开销
4. **线程安全**：当前实现不是线程安全的，多线程环境下需要额外处理

## 扩展功能

可以根据需要扩展以下功能：

1. **日志轮转**：自动清理旧日志文件
2. **压缩存储**：压缩大型日志文件
3. **实时监控**：添加实时日志查看功能
4. **统计分析**：集成更多分析工具 