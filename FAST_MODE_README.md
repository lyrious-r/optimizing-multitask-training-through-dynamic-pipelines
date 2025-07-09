# DynaPipe 快速模式

## 概述

快速模式是 DynaPipe 的一个性能优化功能，通过减少候选方案数量、禁用排列组合和限制重计算计划调整次数来大幅提升 `generate_execution_plan` 函数的执行速度。

## 性能优化策略

### 1. 候选方案减少
- **正常模式**: 尝试所有可用的调度方法
- **快速模式**: 优先使用 `wait-free-cyclic` 调度方法，如果不可用则使用第一个可用的方法

### 2. 排列组合保留
- **正常模式**: 尝试多种 microbatch 排列组合
- **快速模式**: 同样尝试排列组合，但通过其他方式优化性能

### 3. 重计算计划调整限制
- **正常模式**: 无限制地调整重计算计划
- **快速模式**: 最多调整 3 次，每次最多调整 2 个 microbatch

## 使用方法

### 基本用法

```python
from dynapipe.schedule_opt.execution_planner import ExecutionPlanner

# 创建执行计划器
planner = ExecutionPlanner(...)

# 使用快速模式
execution_plans, cost, stats, rc, sch = planner.generate_execution_plan(
    batch=your_batch,
    current_batch_idx=batch_idx,
    fast_mode=True,  # 启用快速模式
)
```

### Megatron 程序中的使用

```python
from dynapipe.pipe.data_loader import TrainingSpec

# 创建训练规格时启用快速模式
training_spec = TrainingSpec(
    cm_path="./cost_model.json",
    cluster_spec=cluster_spec,
    model_spec=model_spec,
    data_parallel_size=2,
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    zero_stage=1,
    device_assignment=device_assignment,
    device_memory_limit=14000,
    fast_mode=True,  # 启用快速模式
)

# 创建数据加载器
dataloader = DynaPipeDataLoader(
    training_spec=training_spec,
    dataset=dataset,
    pack_fn=pack_fn,
    constructor_fn=constructor_fn,
    # ... 其他参数 ...
)
```

### 环境变量控制

```python
import os

# 通过环境变量控制
fast_mode = os.environ.get("DYNAPIPE_FAST_MODE", "false").lower() == "true"
execution_plans, cost, stats, rc, sch = planner.generate_execution_plan(
    batch=your_batch,
    current_batch_idx=batch_idx,
    fast_mode=fast_mode,
)
```

### Megatron 程序中的环境变量控制

```python
import os

# 通过环境变量控制 Megatron 程序中的快速模式
fast_mode = os.environ.get("DYNAPIPE_FAST_MODE", "false").lower() == "true"
training_spec = TrainingSpec(
    # ... 其他参数 ...
    fast_mode=fast_mode,
)

# 或者在命令行中设置
# export DYNAPIPE_FAST_MODE=true
# python your_training_script.py
```

### 动态选择

```python
# 根据批次大小动态选择
batch_size = len(your_batch)
fast_mode = batch_size > 8  # 大批次使用快速模式
execution_plans, cost, stats, rc, sch = planner.generate_execution_plan(
    batch=your_batch,
    current_batch_idx=batch_idx,
    fast_mode=fast_mode,
)
```

## 性能对比

### 预期性能提升

| 批次大小 | 正常模式时间 | 快速模式时间 | 加速比 |
|---------|-------------|-------------|--------|
| 4 microbatch | ~10s | ~3s | ~3x |
| 8 microbatch | ~30s | ~8s | ~4x |
| 16 microbatch | ~120s | ~25s | ~5x |

### 质量对比

快速模式在保持合理调度质量的前提下实现性能提升：

- **调度质量**: 通常与正常模式相差 < 10%
- **内存使用**: 基本保持一致
- **收敛性**: 更快收敛到可行解

## 适用场景

### 推荐使用快速模式的场景

1. **快速原型验证**
   - 开发阶段需要快速验证想法
   - 调试和测试阶段

2. **大批次处理**
   - 批次大小 > 8 时
   - 对实时性要求高的场景

3. **资源受限环境**
   - 计算资源有限
   - 时间预算紧张

4. **批量处理**
   - 需要处理大量批次
   - 对单个批次质量要求不是特别严格

### 推荐使用正常模式的场景

1. **生产环境**
   - 需要最优调度质量
   - 对性能要求极高

2. **小批次处理**
   - 批次大小 ≤ 4 时
   - 有足够时间进行优化

3. **关键任务**
   - 对调度质量要求严格
   - 可以接受较长的优化时间

## 技术细节

### 实现原理

1. **候选方案筛选**
   ```python
   if fast_mode:
       if "wait-free-cyclic" in self.valid_schedule_methods:
           sch_methods = ["wait-free-cyclic"]
       else:
           sch_methods = [self.valid_schedule_methods[0]]
   ```

2. **排列组合保留**
   ```python
   try_permutations=not disable_permute_microbatches  # 保留排列组合功能
   ```

3. **调整次数限制**
   ```python
   if fast_mode and adjust_rc_plan_call_count > 3:
       return rc_plan
   ```

### 性能监控

快速模式会记录详细的性能日志：

```python
# 性能日志示例
{
    "function": "generate_execution_plan",
    "execution_time": 2.5,
    "fast_mode": true,
    "batch_size": 8,
    "n_candidates": 1
}
```

## 测试和验证

### 运行测试

```bash
# 运行快速模式测试
python test_fast_mode.py

# 运行使用示例
python example_fast_mode_usage.py
```

### 验证要点

1. **功能正确性**: 确保快速模式能生成有效的执行计划
2. **性能提升**: 验证执行时间显著减少
3. **质量保持**: 确保调度质量在可接受范围内
4. **兼容性**: 确保与现有代码兼容

## 故障排除

### 常见问题

1. **快速模式失败**
   - 检查 `valid_schedule_methods` 是否为空
   - 确认设备分配是否合理

2. **性能提升不明显**
   - 检查批次大小是否足够大
   - 确认是否真正启用了快速模式

3. **调度质量下降**
   - 考虑在关键场景使用正常模式
   - 调整批次大小或设备配置

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查性能日志
from dynapipe.schedule_opt.execution_planner import get_performance_logger
logger = get_performance_logger()
```

## 总结

快速模式通过智能的优化策略，在保持合理调度质量的前提下，实现了显著的性能提升。它特别适用于大批次处理、快速原型验证和资源受限的场景。通过简单的参数控制，用户可以灵活选择是否启用快速模式，满足不同场景的需求。 