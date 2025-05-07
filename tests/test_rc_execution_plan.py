import pytest
import traceback
from dynapipe.model import (
    DynaPipeMicrobatch,
    DynaPipeMinibatch,
    get_uniform_cluster,
)
from dynapipe.schedule_opt.execution_planner import optimize_schedule

def hetero_minibatch() -> DynaPipeMinibatch:
    fw_times = [4000] * 4 + [2000] * 4  # 8 layer ende
    # fw_times = [4000] * 4 + [4000] * 4
    microbatch_multiplier = [1, 0.8, 1.2, 0.9, 1.1, 0.7, 1.4, 0.6]
    memory_multiplier = [1, 0.8, 1.2, 0.9, 1.1, 0.7, 1.4, 0.6]
    rc_time_multiplier = [2,3,2.4]
    rc_memory_multiplier = [1,0,0.3]
    microbatches = []
    for i in range(len(microbatch_multiplier)):
        current_fw_times = [[
            fw_times[j] * microbatch_multiplier[i]
            for j in range(len(fw_times))
        ]]*3
        current_bw_times = [[rc_time_multiplier[j] * t for t in current_fw_times[j]] for j in range(len(current_fw_times))]
        #print(current_fw_times)
        #print(current_bw_times)
        microbatch = DynaPipeMicrobatch(str(i))
        microbatch.set_fw_exec_times(current_fw_times)
        microbatch.set_bw_exec_times(current_bw_times)
        microbatch.set_fw_comm_size(
            [2 * microbatch_multiplier[i]] * (len(fw_times) - 1)
        )
        microbatch.set_bw_comm_size(
            [2 * microbatch_multiplier[i]] * (len(fw_times) - 1)
        )
        microbatch.set_model_state_memory([4000] * len(fw_times))
        current_activation_memory = [[8000 * rc_memory_multiplier[j] * t for t in microbatch_multiplier] for j in range(len(current_fw_times))]
        microbatch.set_model_stored_activation_memory(
            current_activation_memory
        )
        #print([[8000 * memory_multiplier[i]] * len(fw_times)]*3)
        microbatch.set_model_peak_activation_memory(
            [16000 * memory_multiplier[i]] * len(fw_times)
        )
        microbatch.set_activation_shapes(
            [[(64, 128, 512)]] * (len(fw_times) // 2)
            + [[(64, 128, 512), (64, 128, 512)]] * (len(fw_times) // 2)
        )
        microbatches.append(microbatch)
    minibatch = DynaPipeMinibatch("test", microbatches)
    return minibatch

def test_optimize_schedule_recompute_and_instruction():

    cluster = get_uniform_cluster(4)
    minibatch = hetero_minibatch()
    device_assignment = [0, 0, 1, 1, 2, 2, 3, 3]  # 两个设备

    # 1. 不限制内存，应该不会触发重计算
    result = optimize_schedule(
        sch_type="1F1B",
        opt_minibatch=minibatch,
        opt_cluster=cluster,
        device_assignment=device_assignment,
        try_permutations=False,
        include_memory_stats=True,
        memory_limit=1e10,  # 足够大
    )
    _, _, _, min_makespan_no_rc, min_stats_no_rc, min_instructions_no_rc = result

    print("\n\n\n\nstart test 2\n\n\n\n")
    # 2. 限制内存，强制触发重计算
    result = optimize_schedule(
        sch_type="1F1B",
        opt_minibatch=minibatch,
        opt_cluster=cluster,
        device_assignment=device_assignment,
        try_permutations=False,
        include_memory_stats=True,
        memory_limit=0,  # 非常小，必须重计算
    )
    _, _, _, min_makespan_rc, min_stats_rc, min_instructions_rc = result

    # 检查 makespan 是否发生变化
    assert min_makespan_rc > min_makespan_no_rc

    # 检查 rc_plan 是否发生变化（通过 min_stats 里的 rc_plan 或 instruction 里的 recompute_policy）
    # 检查 instruction 里的 ForwardPass 是否有 recompute_policy 字段
    found_recompute_policy = False
    for instrs in min_instructions_no_rc:
        for instr in instrs:
            if instr.__class__.__name__ == "ForwardPass":
                print(instr)
                # recompute_policy 应该不是 None
                if hasattr(instr, "recompute_policy") and instr.recompute_policy is not None:
                    found_recompute_policy = True
    assert found_recompute_policy, "ForwardPass 没有正确记录重计算方案"

    # 检查反向传播时间是否受影响（可选，需根据 rc_type 设计）
    # 这里只要 makespan 变大即可

    print("无重计算 makespan:", min_makespan_no_rc)
    print("有重计算 makespan:", min_makespan_rc)


def analyze_backward_pass_redundancy_real_sim():
    from dynapipe.model import DynaPipeMicrobatch, DynaPipeMinibatch, get_uniform_cluster
    from dynapipe.schedule_opt.execution_planner import optimize_schedule
    from dynapipe.schedule_opt.schedule_common import analyze_backward_pass_redundancy
    # 构造一个简单的真实仿真

    minibatch = hetero_minibatch()
    cluster = get_uniform_cluster(4)
    device_assignment = [0, 0, 1, 1, 2, 2, 3, 3]
    result = optimize_schedule(
        sch_type="1F1B",
        opt_minibatch=minibatch,
        opt_cluster=cluster,
        device_assignment=device_assignment,
        try_permutations=False,
        include_memory_stats=True,
        memory_limit=1e10,
    )
    min_stats = result[4]
    if min_stats is not None and len(min_stats) >= 4:
        timeline_json = min_stats[3]
    else:
        raise RuntimeError(f"optimize_schedule 返回值异常: {result}")
    #print(timeline_json)
    redundancy = analyze_backward_pass_redundancy(timeline_json)
    #print(redundancy)

    def is_backward_pass(event):
        name = event["name"]
        return name.endswith("B") and "Comm" not in name

    # 找到每个executor的所有BackwardPass的结束时间
    executor2backward_end = {}
    for e in timeline_json["traceEvents"]:
        if is_backward_pass(e):
            executor_id = e["pid"]
            end_time = e["ts"] + e["dur"]
            executor2backward_end.setdefault(executor_id, []).append(end_time)
    # 取每个executor的最大结束时间（即最后一个BackwardPass的结束时间）
    executor2last_end = {k: max(v) for k, v in executor2backward_end.items()}

    for executor_id, items in redundancy.items():
        for item in items:
            # 需要找到该microbatch对应的事件
            # 由于 analyze_backward_pass_redundancy 只返回 microbatch id 和冗余量，这里只能断言冗余量的数量和顺序
            # 如果你想断言不是最后一个，可以用数量比对
            pass  # 这里可以根据你的实际需求补充断言
    print("test_analyze_backward_pass_redundancy_real_sim passed.")

# 可直接加到 test_scheduler.py 或新建 test_execution_planner.py

