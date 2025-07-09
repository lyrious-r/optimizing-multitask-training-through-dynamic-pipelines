#!/usr/bin/env python3
"""
快速模式使用示例

这个示例展示了如何在现有的 DynaPipe 代码中使用快速模式来大幅提升性能。
"""

import time
import logging
from dynapipe.schedule_opt.execution_planner import ExecutionPlanner
from dynapipe.model import DynaPipeCluster, TransformerModelSpec
from dynapipe.data_opt.cost_models import ProfileBasedCostModelWithRC

def example_usage():
    """展示快速模式的使用方法"""
    
    print("=" * 70)
    print("DynaPipe 快速模式使用示例")
    print("=" * 70)
    
    # 1. 基本设置
    print("\n1. 基本设置")
    model_spec = TransformerModelSpec(
        n_encoder_layers=12,
        n_decoder_layers=12,
        hidden_dim=768,
        n_heads=12,
        vocab_size=50257,
        max_seqlen=512,
    )
    
    cluster_spec = DynaPipeCluster(
        n_devices=4,
        device_memory=16000,
        device_bandwidth=12.0,
    )
    
    device_assignment = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    device_memory_limit = 14000
    
    cost_model = ProfileBasedCostModelWithRC()
    
    # 2. 创建执行计划器
    print("\n2. 创建执行计划器")
    planner = ExecutionPlanner(
        cluster_spec=cluster_spec,
        model_spec=model_spec,
        device_assignment=device_assignment,
        device_memory_limit=device_memory_limit,
        cost_model=cost_model,
    )
    
    # 3. 测试批次
    print("\n3. 测试批次")
    test_batches = [
        [(4, 512, 128), (4, 512, 128), (4, 512, 128), (4, 512, 128)],  # 小批次
        [(8, 512, 128), (8, 512, 128), (8, 512, 128), (8, 512, 128)],  # 中等批次
        [(16, 512, 128), (16, 512, 128), (16, 512, 128), (16, 512, 128)],  # 大批次
    ]
    
    # 4. 性能对比测试
    print("\n4. 性能对比测试")
    print("-" * 50)
    
    for i, batch in enumerate(test_batches):
        print(f"\n批次 {i+1}: {len(batch)} 个 microbatch")
        
        # 正常模式
        start_time = time.time()
        try:
            execution_plans, cost, stats, rc, sch = planner.generate_execution_plan(
                batch=batch,
                current_batch_idx=i,
                fast_mode=False,  # 正常模式
            )
            normal_time = time.time() - start_time
            normal_cost = cost
            print(f"   正常模式: {normal_time:.2f}s, 成本: {normal_cost:.2f}")
        except Exception as e:
            print(f"   正常模式失败: {e}")
            normal_time = float('inf')
            normal_cost = float('inf')
        
        # 快速模式
        start_time = time.time()
        try:
            execution_plans, cost, stats, rc, sch = planner.generate_execution_plan(
                batch=batch,
                current_batch_idx=i,
                fast_mode=True,  # 快速模式
            )
            fast_time = time.time() - start_time
            fast_cost = cost
            print(f"   快速模式: {fast_time:.2f}s, 成本: {fast_cost:.2f}")
        except Exception as e:
            print(f"   快速模式失败: {e}")
            fast_time = float('inf')
            fast_cost = float('inf')
        
        # 性能对比
        if normal_time != float('inf') and fast_time != float('inf'):
            speedup = normal_time / fast_time
            cost_diff = abs(normal_cost - fast_cost) / normal_cost * 100
            print(f"   加速比: {speedup:.2f}x, 成本差异: {cost_diff:.1f}%")
        else:
            print("   无法计算性能对比")
    
    # 5. 使用建议
    print("\n5. 使用建议")
    print("-" * 50)
    print("✓ 快速模式适用于以下场景:")
print("  - 需要快速原型验证")
print("  - 大批次处理时性能要求高")
print("  - 对调度质量要求不是特别严格")
print("  - 开发调试阶段")
print("  - 需要保持排列组合优化")
    print("\n✓ 正常模式适用于以下场景:")
    print("  - 生产环境需要最优调度")
    print("  - 对性能要求极高")
    print("  - 有足够的时间进行优化")
    
    # 6. 代码集成示例
    print("\n6. 代码集成示例")
    print("-" * 50)
    print("""
# 在你的代码中这样使用:

# 方式1: 通过参数控制
execution_plans, cost, stats, rc, sch = planner.generate_execution_plan(
    batch=your_batch,
    current_batch_idx=batch_idx,
    fast_mode=True,  # 启用快速模式
)

# 方式2: 通过环境变量控制
import os
fast_mode = os.environ.get("DYNAPIPE_FAST_MODE", "false").lower() == "true"
execution_plans, cost, stats, rc, sch = planner.generate_execution_plan(
    batch=your_batch,
    current_batch_idx=batch_idx,
    fast_mode=fast_mode,
)

# 方式3: 根据批次大小动态选择
batch_size = len(your_batch)
fast_mode = batch_size > 8  # 大批次使用快速模式
execution_plans, cost, stats, rc, sch = planner.generate_execution_plan(
    batch=your_batch,
    current_batch_idx=batch_idx,
    fast_mode=fast_mode,
)
""")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    example_usage() 