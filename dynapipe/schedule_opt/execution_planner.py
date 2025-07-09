# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import json
import logging
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from dynapipe.data_opt.cost_models import ProfileBasedCostModelWithRC
from dynapipe.model import (
    DynaPipeCluster,
    DynaPipeMicrobatch,
    DynaPipeMinibatch,
    TransformerModelSpec,
    get_simulator,
)
from dynapipe.pipe.instruction_optimizer import InstructionOptimizer
from dynapipe.pipe.instructions import (
    ExecutionPlan,
    PipeInstruction,
    name_to_recompute_method,
)
from dynapipe.pipe.utils import validate_device_assignment
from dynapipe.utils.memory_utils import get_transformer_output_memory
from dynapipe.schedule_opt.schedule_common import analyze_backward_pass_redundancy

# 性能日志系统
class PerformanceLogger:
    def __init__(self, log_dir: str = "./performance_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("PerformanceLogger")
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = os.path.join(log_dir, "performance.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # 创建rc_plan日志文件
        self.rc_plan_file = os.path.join(log_dir, "rc_plans.json")
        
    def log_execution_time(self, function_name: str, execution_time: float, **kwargs):
        """记录函数执行时间"""
        log_entry = {
            "function": function_name,
            "execution_time": execution_time,
            "timestamp": time.time(),
            **kwargs
        }
        self.logger.info(f"EXECUTION_TIME: {json.dumps(log_entry)}")
        
    def log_rc_plan(self, batch_idx: int, rc_plan: List, **kwargs):
        """记录rc_plan方案"""
        log_entry = {
            "batch_idx": batch_idx,
            "rc_plan": rc_plan,
            "timestamp": time.time(),
            **kwargs
        }
        
        # 读取现有的rc_plan日志
        existing_logs = []
        if os.path.exists(self.rc_plan_file):
            try:
                with open(self.rc_plan_file, 'r') as f:
                    existing_logs = json.load(f)
            except:
                existing_logs = []
        
        # 添加新的日志条目
        existing_logs.append(log_entry)
        
        # 写入文件
        with open(self.rc_plan_file, 'w') as f:
            json.dump(existing_logs, f, indent=2)

# 全局性能日志器实例
_performance_logger = None

def get_performance_logger(log_dir: str = None) -> PerformanceLogger:
    """获取性能日志器实例"""
    global _performance_logger
    if _performance_logger is None:
        if log_dir is None:
            log_dir = os.environ.get("DYNAPIPE_PERFORMANCE_LOG_DIR", "./performance_logs")
        _performance_logger = PerformanceLogger(log_dir)
    return _performance_logger

def optimize_schedule(
    sch_type: str,
    opt_minibatch: DynaPipeMinibatch,
    opt_cluster: DynaPipeCluster,
    device_assignment: List[int],
    try_permutations=True,
    perm_clusters=None,
    perm_cluster_algo="kmeans",
    include_memory_stats=False,
    progress_bar=False,
    memory_limit=float("inf"),
    disable_scheduler_memory_limit=False,
    max_otf_microbatches=None,
    raise_on_oom=True,
    rc_type: Optional[str] = None, # NOT USED
    logger: Optional[logging.Logger] = None,
    fast_mode=False,  # 新增快速模式参数
):
    # 记录optimize_schedule开始时间
    perf_logger = get_performance_logger()
    start_time = time.time()
    
    # 添加adjust_rc_plan执行次数计数器
    adjust_rc_plan_call_count = 0
    
    if try_permutations:  # 保留排列组合功能
        if perm_clusters is None:
            if len(opt_minibatch.microbatches) > 20:
                perm_clusters = 3
            else:
                perm_clusters = 4
        if len(opt_minibatch.microbatches) > perm_clusters:
            mb_vectors = []
            for mb in opt_minibatch.microbatches:
                # use fw and bw time as features
                mb_vectors.append(
                    [
                        mb.fw_exec_times[0][0],
                        mb.fw_exec_times[0][-1],
                        mb.bw_exec_times[0][0],
                        mb.bw_exec_times[0][-1],
                    ]
                )
            mb_vectors = np.array(mb_vectors)
            if perm_cluster_algo == "kmeans":
                cluster = KMeans(
                    perm_clusters,
                    random_state=0,
                    n_init="auto",
                ).fit(mb_vectors)
            elif perm_cluster_algo == "agglomerative":
                cluster = AgglomerativeClustering(
                    perm_clusters,
                    linkage="complete",
                ).fit(mb_vectors)
            mb_labels = list(cluster.labels_)
            n_clusters = max(mb_labels) + 1
            assert n_clusters <= perm_clusters
            mb_groups = [[] for _ in range(n_clusters)]
            mb_idx2group = {}
            for i, label in enumerate(mb_labels):
                mb_groups[label].append(i)
                mb_idx2group[i] = label
            result_premutations = []
            for perm in itertools.permutations(range(len(mb_groups))):
                # generate a random permutation for each group
                mb_random_perm_per_label = {}
                for label, mb_indices in enumerate(mb_groups):
                    shuffled_indices = np.random.permutation(mb_indices)
                    mb_random_perm_per_label[label] = list(shuffled_indices)
                reconstructed_perm = []
                for label in perm:
                    reconstructed_perm.extend(mb_random_perm_per_label[label])
                result_premutations.append(reconstructed_perm)
            permutations = result_premutations
        else:
            permutations = list(
                itertools.permutations(range(len(opt_minibatch.microbatches)))
            )
    else:
        permutations = []
    # always try the original order
    permutations.append(list(range(len(opt_minibatch.microbatches))))

    def _run_schedules(scheduler_memory_limit):
        nonlocal adjust_rc_plan_call_count  # 使用外部计数器
        max_makespan = 0.0
        max_stats = None
        max_instructions = []
        min_rc_plan = []
        min_makespan = float("inf")
        min_stats = None
        min_instructions = []
        if progress_bar:
            from tqdm import tqdm

            iterator = tqdm(permutations)
        else:
            iterator = permutations
        debug_json = None
        mem_for_perms = []

        def adjust_rc_plan(rc_plan, peak_memory, trace_events, memory_limit, topk=1):
            # 快速模式下使用更激进的调整策略
            if fast_mode:
                topk = min(topk, 2)  # 快速模式下最多调整2个microbatch
            """
            只对 peak_memory 超过 memory_limit 的 compute executor, 调整其冗余最多的 topk 个 microbatch 的 rc_type。
            - 如果 redundancy > 1, new_rc_type 直接设为 1。
            - rc_type == 1 的 microbatch 不参与冗余分析和调整。
            - 冗余分析结果为空时不做任何调整。
            - 快速模式下减少调整次数。
            其它不变。
            """
            nonlocal adjust_rc_plan_call_count  # 使用外部计数器
            adjust_rc_plan_call_count += 1  # 增加调用次数
            
            # 快速模式下限制调整次数
            if fast_mode and adjust_rc_plan_call_count > 3:  # 最多调整3次
                return rc_plan
            
            # 记录adjust_rc_plan开始时间
            adjust_start_time = time.time()
            
            import re
            # 1. 找出超限的 compute executor id
            over_limit_executors = set()
            for full_name, mem in peak_memory.items():
                if "Compute" in full_name and mem > memory_limit:
                    m = re.search(r"Executor (\d+)", full_name)
                    if m:
                        executor_id = int(m.group(1))
                        over_limit_executors.add(executor_id)
            # 2. 冗余分析
            redundancy = analyze_backward_pass_redundancy(trace_events,rc_plan,device_assignment)
            # 3. 记录需要调整的 (microbatch, executor) -> new_rc_type
            to_adjust = dict()  # (mb, executor) -> new_rc_type
            for executor_id, items in redundancy.items():
                if executor_id not in over_limit_executors:
                    continue
                count = 0
                for item in items:
                    if count >= topk:
                        break
                    # 跳过 rc_type==1 的 microbatch（后面判断）
                    # 冗余大于1直接设为1，否则按原逻辑
                    if item["redundancy"] >= 0.5:
                        to_adjust[(item["microbatch"], executor_id)] = 1
                    else:
                        to_adjust[(item["microbatch"], executor_id)] = 2
                    count += 1
            # 4. 遍历rc_plan，调整对应microbatch在该executor的rc_type
            new_rc_plan = []
            for mb_idx, mb_plan in enumerate(rc_plan):
                new_mb_plan = []
                for layer_idx, (start, end, rc_type) in enumerate(mb_plan):
                    executor_id = device_assignment[layer_idx]
                    # rc_type==1 的不参与冗余调整
                    if rc_type == 1:
                        new_mb_plan.append((start, end, rc_type))
                        continue
                    key = (mb_idx, executor_id)
                    if key in to_adjust:
                        if rc_type == 0:
                            new_rc_type = to_adjust[key]
                        else:
                            new_rc_type = 1
                        new_mb_plan.append((start, end, new_rc_type))
                    else:
                        new_mb_plan.append((start, end, rc_type))
                new_rc_plan.append(new_mb_plan)
            
            # 记录adjust_rc_plan执行时间
            adjust_execution_time = time.time() - adjust_start_time
            perf_logger.log_execution_time(
                "adjust_rc_plan", 
                adjust_execution_time,
                memory_limit=memory_limit,
                topk=topk,
                over_limit_executors_count=len(over_limit_executors),
                call_count=adjust_rc_plan_call_count  # 记录当前调用次数
            )
            
            return new_rc_plan


        for perm in iterator:
            permuted_minibatch = opt_minibatch.permute_microbatches(perm)
            n_stages = len(permuted_minibatch.microbatches[0].fw_exec_times[0])  # 获取总stage数
            rc_plan = [[(i, i+1, 0) for i in range(n_stages)] for _ in range(len(permuted_minibatch.microbatches))]  # [(start_stage, end_stage, rc_type)] 0表示none
            while True:
                #print("########################## new iter #####################")
                # get simulator
                simulator = get_simulator(
                    sch_type,
                    permuted_minibatch,
                    opt_cluster,
                    device_assignment,
                    rc_plan=rc_plan,
                    include_memory_stats=include_memory_stats,
                    memory_limit=scheduler_memory_limit,
                    max_otf_microbatches=max_otf_microbatches,
                    logger=logger,
                )
                timeline_json = simulator.schedule()
                instructions = simulator.get_instructions()
                peak_memory = simulator.get_executor_peak_memory()
                max_memory_device = -1
                max_device_memory = -1
                for device, memory in peak_memory.items():
                    if memory > max_device_memory:
                        max_memory_device = device
                        max_device_memory = memory
                makespan = simulator.get_makespan()
                if makespan is None:
                    continue
                makespan = makespan / 1000.0
                debug_json = timeline_json
                mem_for_perms.append(max_device_memory)
                # 检查内存和时间约束
                if max_device_memory > memory_limit:
                    # 分析每个stage的内存使用情况,修改rc_plan
                    new_rc_plan = adjust_rc_plan(rc_plan, peak_memory, timeline_json, memory_limit)
                    #print("new plan generated")
                    if new_rc_plan == rc_plan:  # 如果无法进一步优化
                        #print("no more optimization")
                        break
                    rc_plan = new_rc_plan
                    continue

                break

            if makespan > max_makespan:
                max_makespan = makespan
                max_stats = (
                    perm,
                    max_device_memory,
                    max_memory_device,
                    timeline_json,
                )
                max_instructions = instructions
            
            if makespan < min_makespan:
                min_makespan = makespan
                min_stats = (
                    perm,
                    max_device_memory,
                    max_memory_device,
                    timeline_json,
                )
                min_instructions = instructions
                min_rc_plan = rc_plan
                
        if logger is not None and max_makespan > 0.0:
            logger.debug(
                "Sched mem limit: {}, Schedule type: {}, "
                "min peak memory: {} MB, makespan: {}.".format(
                    scheduler_memory_limit,
                    sch_type,
                    min(mem_for_perms),
                    min_makespan,
                )
            )
        return (
            max_makespan,
            max_stats,
            max_instructions,
            min_makespan,
            min_stats,
            min_instructions,
            min_rc_plan,  # 返回最终的rc_plan
            debug_json,
            mem_for_perms,
        )

    # first try without setting memory limit on scheduler
    # (i.e. see if there exist a feasible permutation)
    (
        max_makespan,
        max_stats,
        max_instructions,
        min_makespan,
        min_stats,
        min_instructions,
        min_rc_plan,
        debug_json,
        mem_for_perms,
    ) = _run_schedules(float("inf"))
    if (
        max_makespan == 0.0
        and sch_type == "wait-free-cyclic"
        and not disable_scheduler_memory_limit
    ):
        # try with scheduler memory limit
        if logger is not None:
            logger.debug("Trying with scheduler memory limit.")
        (
            max_makespan,
            max_stats,
            max_instructions,
            min_makespan,
            min_stats,
            min_instructions,
            min_rc_plan,
            debug_json,
            mem_for_perms,
        ) = _run_schedules(memory_limit)
    if max_makespan == 0.0 and raise_on_oom:
        # with open("./test_memory.json", "w") as f:
        #     json.dump(debug_json, f)
        raise RuntimeError(
            "No feasible schedule within memory limit found. "
            "Memory consumption for different permutations: "
            "min: {}, max: {}.".format(
                [] if not mem_for_perms else min(mem_for_perms),
                [] if not mem_for_perms else max(mem_for_perms),
            )
        )

    # 记录optimize_schedule执行时间
    execution_time = time.time() - start_time
    perf_logger.log_execution_time(
        "optimize_schedule", 
        execution_time,
        sch_type=sch_type,
        n_microbatches=len(opt_minibatch.microbatches),
        memory_limit=memory_limit,
        permutations_count=len(permutations),
        adjust_rc_plan_call_count=adjust_rc_plan_call_count  # 记录总的adjust_rc_plan调用次数
    )

    return (
        max_makespan,
        max_stats,
        max_instructions,
        min_makespan,
        min_stats,
        min_instructions,
        min_rc_plan,
    )


def construct_minibatch_spec(
    model_spec: TransformerModelSpec,
    cost_model: ProfileBasedCostModelWithRC,
    minibatch: List[Tuple[int, int, int]],
    dp_size: int = 1,
    tp_size: int = 1,
    zero_stage: int = 0,
    minibatch_idx: Optional[int] = None,
    name="microbatch",
):
    microbatches = []
    for microbatch_idx, (mbsize, input_seqlen, target_seqlen) in enumerate(minibatch):
        # sanity check
        if model_spec.n_decoder_layers == 0 and target_seqlen != 0:
            raise ValueError(
                "Target sequence length must be 0 if there are "
                "no decoder layers."
            )
        if target_seqlen == 0 and model_spec.n_decoder_layers > 0:
            raise ValueError(
                "Target sequence length cannot be 0 if there are "
                "decoder layers."
            )
        mb = DynaPipeMicrobatch(str(microbatch_idx))

        def _get_cost(stage_name, seqlen):
            return (
                cost_model.get_cost(
                    tp_size,
                    stage_name,
                    seqlen,
                    mbsize,
                )
                * 1000
            )

        def _get_stored_activation(stage_name, seqlen):
            return cost_model.get_stored_activation(
                tp_size,
                stage_name,
                seqlen,
                mbsize,
            )

        def _get_peak_activation(stage_name, seqlen):
            return cost_model.get_peak_activation(
                tp_size,
                "none",  # 使用默认的none类型
                stage_name,
                seqlen,
                mbsize,
            )

        # 获取执行时间和存储激活内存
        enc_fw_times = _get_cost("Encoder FW", input_seqlen)
        enc_bw_times = _get_cost("Encoder BW", input_seqlen)
        enc_stored_activation = _get_stored_activation("Encoder", input_seqlen)

        # 后处理时间
        if target_seqlen > 0:
            dec_fw_times = _get_cost("Decoder FW", (input_seqlen, target_seqlen))
            dec_bw_times = _get_cost("Decoder BW", (input_seqlen, target_seqlen))
            dec_stored_activation = _get_stored_activation("Decoder", (input_seqlen, target_seqlen))
            dec_postprocess_fw_times = _get_cost("Postprocess FW", target_seqlen)
            dec_postprocess_bw_times = _get_cost("Postprocess BW", target_seqlen)
            enc_postprocess_fw_times = [0, 0, 0]
            enc_postprocess_bw_times = [0, 0, 0]
        else:
            dec_fw_times = [0, 0, 0]
            dec_bw_times = [0, 0, 0]
            dec_stored_activation = [0, 0, 0]
            dec_postprocess_fw_times = [0, 0, 0]
            dec_postprocess_bw_times = [0, 0, 0]
            enc_postprocess_fw_times = _get_cost("Postprocess FW", input_seqlen)
            enc_postprocess_bw_times = _get_cost("Postprocess BW", input_seqlen)

        # 为每种重计算类型构建执行时间列表
        fw_times_list = []
        bw_times_list = []
        stored_activation_list = []

        for rc_idx in range(3):  # 0=none, 1=full, 2=selective
            fw_times = ([enc_fw_times[rc_idx]] * (model_spec.n_encoder_layers - 1) +
                       [enc_fw_times[rc_idx] + enc_postprocess_fw_times[rc_idx]] +
                       [dec_fw_times[rc_idx]] * max(0, model_spec.n_decoder_layers - 1) +
                       ([dec_fw_times[rc_idx] + dec_postprocess_fw_times[rc_idx]]
                        if target_seqlen > 0 else []))
            fw_times_list.append(fw_times)
            
            bw_times = ([dec_bw_times[rc_idx] + dec_postprocess_bw_times[rc_idx]]
                       if target_seqlen > 0 else []) + \
                      [dec_bw_times[rc_idx]] * max(0, model_spec.n_decoder_layers - 1) + \
                      [enc_bw_times[rc_idx] + enc_postprocess_bw_times[rc_idx]] + \
                      [enc_bw_times[rc_idx]] * (model_spec.n_encoder_layers - 1)
            bw_times_list.append(bw_times)
            
            stored_activation = [enc_stored_activation[rc_idx]] * model_spec.n_encoder_layers + \
                              [dec_stored_activation[rc_idx]] * model_spec.n_decoder_layers
            stored_activation_list.append(stored_activation)

        # 设置执行时间和存储激活内存
        mb.set_fw_exec_times(fw_times_list)
        mb.set_bw_exec_times(bw_times_list)
        mb.set_model_stored_activation_memory(stored_activation_list)

        # 获取峰值激活内存（不区分重计算类型）
        enc_peak_activation_memory = _get_peak_activation("Encoder", input_seqlen)
        if target_seqlen > 0:
            dec_peak_activation_memory = _get_peak_activation(
                "Decoder", (input_seqlen, target_seqlen)
            )
        else:
            dec_peak_activation_memory = 0

        # 设置其他不受重计算影响的属性
        mb.set_model_peak_activation_memory(
            [enc_peak_activation_memory] * model_spec.n_encoder_layers +
            [dec_peak_activation_memory] * model_spec.n_decoder_layers
        )

        # 设置通信大小和模型状态内存（使用默认的none类型）
        emb_model_state_memory = cost_model.get_model_state(
            tp_size,
            "none",
            "Embedding",
            n_shards=dp_size,
            zero_stage=zero_stage,
        )
        enc_model_state_memory = cost_model.get_model_state(
            tp_size,
            "none",
            "Encoder",
            n_shards=dp_size,
            zero_stage=zero_stage,
        )
        if target_seqlen > 0:
            dec_model_state_memory = cost_model.get_model_state(
                tp_size,
                "none",
                "Decoder",
                n_shards=dp_size,
                zero_stage=zero_stage,
            )
        else:
            dec_model_state_memory = 0

        enc_model_output_memory = get_transformer_output_memory(
            input_seqlen, mbsize, model_spec.hidden_dim, bytes_per_element=2
        )
        if target_seqlen > 0:
            dec_model_output_memory = get_transformer_output_memory(
                target_seqlen,
                mbsize,
                model_spec.hidden_dim,
                bytes_per_element=2,
            )
        else:
            dec_model_output_memory = 0
        # sanity check
        stats = [
            enc_fw_times,
            enc_bw_times,
            dec_fw_times,
            dec_bw_times,
            emb_model_state_memory,
            enc_stored_activation,
            dec_stored_activation,
            enc_peak_activation_memory,
            dec_peak_activation_memory,
            enc_model_state_memory,
            dec_model_state_memory,
        ]
        stats_names = [
            "enc_fw_time",
            "enc_bw_time",
            "dec_fw_time",
            "dec_bw_time",
            "emb_model_state_memory",
            "enc_stored_activation_memory",
            "dec_stored_activation_memory",
            "enc_peak_activation_memory",
            "dec_peak_activation_memory",
            "enc_model_state_memory",
            "dec_model_state_memory",
        ]
        for s, s_name in zip(stats, stats_names):
            def is_invalid(val):
                return val is None or math.isnan(val) or math.isinf(val)
            if isinstance(s, (list, tuple)):
                if any(is_invalid(x) for x in s):
                    return None
            else:
                if is_invalid(s):
                    return None


        mb.set_fw_comm_size(
            [enc_model_output_memory]
            * (model_spec.n_encoder_layers - (1 if target_seqlen == 0 else 0))
            + [enc_model_output_memory + dec_model_output_memory]
            * max(0, model_spec.n_decoder_layers - 1)
        )
        mb.set_bw_comm_size(
            [enc_model_output_memory + dec_model_output_memory]
            * model_spec.n_decoder_layers
            + [enc_model_output_memory] * (model_spec.n_encoder_layers - 1)
        )

        # 设置模型状态内存
        mb.set_model_state_memory(
            [emb_model_state_memory + enc_model_state_memory]
            + [enc_model_state_memory] * (model_spec.n_encoder_layers - 1)
            + (
                [emb_model_state_memory + dec_model_state_memory]
                if target_seqlen > 0
                else []
            )
            + [dec_model_state_memory]
            * max(0, model_spec.n_decoder_layers - 1)
        )

        # 设置激活形状
        mb.set_activation_shapes(
            [[(mbsize, input_seqlen, model_spec.hidden_dim)]]
            * model_spec.n_encoder_layers
            + [
                [
                    (mbsize, input_seqlen, model_spec.hidden_dim),
                    (mbsize, target_seqlen, model_spec.hidden_dim),
                ]
            ]
            * model_spec.n_decoder_layers
        )

        # 检查所有属性是否设置完成
        mb.check_all_set()
        microbatches.append(mb)

    minibatch_spec = DynaPipeMinibatch(name, microbatches)
    return minibatch_spec


class ExecutionPlanner:
    def __init__(
        self,
        cluster_spec: DynaPipeCluster,
        model_spec: TransformerModelSpec,
        device_assignment: List[int],
        device_memory_limit: int,
        cost_model: ProfileBasedCostModelWithRC,
        dp_size: int = 1,
        tp_size: int = 1,
        zero_stage: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cluster_spec = cluster_spec
        self.model_spec = model_spec
        self.cost_model = cost_model
        self.device_assignment = device_assignment
        self.n_devices = max(device_assignment) + 1
        self.device_memory_limit = device_memory_limit
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.zero_stage = zero_stage
        self.logger = logger
        (
            self.device_assignment_type,
            self.valid_schedule_methods,
            self.n_layers_per_stage,
            self.n_chunks_per_device,
        ) = validate_device_assignment(
            model_spec, cluster_spec, self.device_assignment
        )

    def _create_candidates(
        self,
        batch: List[Tuple[int, int, int]],
        schedule_method="dynamic",
        rc_type=None,
        fast_mode=False,  # 新增快速模式参数
    ):
        # 保持原有的 rc_type 处理逻辑
        if rc_type is not None:
            if not isinstance(rc_type, list):
                available_rc_types = [rc_type]
            else:
                available_rc_types = rc_type
        else:
            available_rc_types = ["none", "selective", "full"]

        # 处理调度方法
        if schedule_method == "dynamic":
            if fast_mode:
                # 快速模式下优先使用 wait-free-cyclic，如果不可用则使用第一个可用的方法
                if "wait-free-cyclic" in self.valid_schedule_methods:
                    sch_methods = ["wait-free-cyclic"]
                elif self.valid_schedule_methods:
                    sch_methods = [self.valid_schedule_methods[0]]
                else:
                    sch_methods = []
            else:
                sch_methods = self.valid_schedule_methods
        else:
            if schedule_method not in self.valid_schedule_methods:
                raise ValueError(
                    "Invalid schedule scheme: "
                    "{} for device assignment: {}".format(
                        schedule_method, self.device_assignment
                    )
                )
            sch_methods = [schedule_method]

        # 生成候选项
        candidates = []
        # 只调用一次 construct_minibatch_spec
        minibatch_spec = construct_minibatch_spec(
            self.model_spec,
            self.cost_model,
            batch,
            dp_size=self.dp_size,
            tp_size=self.tp_size,
            zero_stage=self.zero_stage,
        )
        if minibatch_spec is not None:
            for sch in sch_methods:
                candidates.append((sch, minibatch_spec))
        return candidates

    def _optimize_instructions(
        self,
        instructions: List[List[PipeInstruction]],
        n_stages: int,
    ):
        # instructions: instructions for each executor
        # Necessary steps to ensure correctness:
        #   1. Add CommunicationFinishInsturctions at appropriate places
        #   2. Allocate buffer slots (not buffer themselves)
        # Potential optimizations:
        #   1. Merge consecutive communication instructions (trade-off)
        #   2. Reschedule communication instructions
        #   3. Pre-allocate buffers to reduce memory fragmentation
        instrs, n_buffers = InstructionOptimizer(
            instructions, n_stages
        ).optimize()
        return instrs, n_buffers

    def generate_execution_plan(
        self,
        batch: List[Tuple[int, int, int]],
        limit_rc_type=None,
        schedule_method="dynamic",
        disable_permute_microbatches=False,
        disable_scheduler_memory_limit=False,
        current_batch_idx=None,
        fast_mode=False,  # 新增快速模式参数
    ):
        # 记录generate_execution_plan开始时间
        perf_logger = get_performance_logger()
        start_time = time.time()
        
        # 快速模式：减少候选方案和排列组合
        candidates = self._create_candidates(
            batch, schedule_method=schedule_method, rc_type=limit_rc_type, fast_mode=fast_mode
        )
        
        best_instrs = None
        best_sch = None
        best_rc = "none"
        best_cost = None
        best_stats = None
        final_rc_plan = None  # 用于记录最终的rc_plan
        
        for schedule_method, minibatch_spec in candidates:
            (
                max_makespan,
                _,
                _,
                min_makespan,
                min_stats,
                min_instructions,
                min_rc_plan,  # 获取返回的rc_plan
            ) = optimize_schedule(
                schedule_method,
                minibatch_spec,
                self.cluster_spec,
                self.device_assignment,
                try_permutations=not disable_permute_microbatches,  # 保留排列组合功能
                include_memory_stats=True,
                progress_bar=False,
                memory_limit=self.device_memory_limit,
                disable_scheduler_memory_limit=disable_scheduler_memory_limit,
                raise_on_oom=False,
                logger=self.logger,
                fast_mode=fast_mode,  # 传递快速模式参数
            )
            if max_makespan < 1e-5:
                # no feasible schedule
                if self.logger:
                    self.logger.debug(
                        "No feasible schedule for batch {} "
                        "using {} and recompute {}".format(
                            current_batch_idx, schedule_method, rc_type
                        )
                    )
                continue
            if best_cost is None or min_makespan < best_cost:
                best_cost = min_makespan
                best_sch = schedule_method
                #best_rc = min_rc_plan
                best_instrs = min_instructions
                best_stats = min_stats
                final_rc_plan = min_rc_plan  # 使用返回的rc_plan
        
        if best_instrs is None:
            raise RuntimeError(
                "No feasible schedule for batch {}.".format(current_batch_idx)
            )
        # get total number of stages
        best_instrs: List[List[PipeInstruction]]
        n_stages = (
            max([instr.stage for instrs in best_instrs for instr in instrs])
            + 1
        )
        assigned_stages_per_executor = []
        for instrs in best_instrs:
            assigned_stages = set()
            for instr in instrs:
                assigned_stages.add(instr.stage)
            assigned_stages = sorted(list(assigned_stages))
            assigned_stages_per_executor.append(assigned_stages)
        # construct execution plan
        if best_cost is None:
            # no feasible schedule
            return None, None, None, None, None
        assert len(best_instrs) == self.n_devices
        # run necessary optimization pass on instructions
        optimized_instrs, n_buffers = self._optimize_instructions(
            best_instrs, n_stages
        )
        execution_plans = [
            ExecutionPlan(
                instr,
                len(batch),
                self.n_devices,
                n_stages,
                i,
                assigned_stages_per_executor[i],
                name_to_recompute_method(best_rc),
                n_buffer,
            )
            for i, (instr, n_buffer) in enumerate(
                zip(optimized_instrs, n_buffers)
            )
        ]
        
        # 记录generate_execution_plan执行时间
        execution_time = time.time() - start_time
        perf_logger.log_execution_time(
            "generate_execution_plan", 
            execution_time,
            batch_size=len(batch),
            current_batch_idx=current_batch_idx,
            schedule_method=schedule_method,
            best_cost=best_cost,
            n_candidates=len(candidates),
            fast_mode=fast_mode  # 记录是否使用快速模式
        )
        
        # 记录最终的rc_plan方案
        if final_rc_plan is not None and current_batch_idx is not None:
            perf_logger.log_rc_plan(
                current_batch_idx,
                final_rc_plan,
                best_cost=best_cost,
                best_schedule_method=best_sch,
                batch_size=len(batch),
                fast_mode=fast_mode  # 记录是否使用快速模式
            )
        
        return execution_plans, best_cost, best_stats, best_rc, best_sch
