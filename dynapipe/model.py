# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

from dynapipe.schedule_opt import get_scheduler_class
from dynapipe.schedule_opt.schedule_common import (
    Scheduler,
    SchedulerMicrobatchSpec,
    SchedulerMinibatchSpec,
)


class RecomputeMethod:
    NONE = 0
    FULL = 1
    SELECTIVE = 2


@dataclass
class TransformerModelSpec:
    # Default setting:
    #  * mlp_hidden_size = 4x hidden_dim
    #  * kv_channels = hidden_dim // num_attn_heads
    #  * use FP16 mixed precision training with Adam optimizer.
    n_encoder_layers: int
    n_decoder_layers: int
    hidden_dim: int
    num_attn_heads: int
    mlp_hidden_dim: Union[None, int] = None
    kv_channels: Union[None, int] = None
    bytes_per_element: int = 2
    optimizer_state_multiplier: int = 12

    def __post_init__(self):
        if self.mlp_hidden_dim is None:
            # if not specified, use the 4x hidden dim as it is the norm
            self.mlp_hidden_dim = self.hidden_dim * 4
        if self.kv_channels is None:
            # if not specified, use the hidden_dim // num_attn_heads
            assert self.hidden_dim % self.num_attn_heads == 0
            self.kv_channels = self.hidden_dim // self.num_attn_heads

    def serialize(self) -> bytes:
        def _serialize_int(x: int):
            return x.to_bytes(4, "little")

        return b"".join(
            [
                _serialize_int(x)
                for x in [
                    self.n_encoder_layers,
                    self.n_decoder_layers,
                    self.hidden_dim,
                    self.num_attn_heads,
                    self.mlp_hidden_dim,
                    self.kv_channels,
                    self.bytes_per_element,
                    self.optimizer_state_multiplier,
                ]
            ]
        )

    @classmethod
    def deserialize(cls, data: bytes):
        def _deserialize_int(data: bytes):
            return int.from_bytes(data, "little")

        return cls(
            *[_deserialize_int(data[i * 4 : (i + 1) * 4]) for i in range(8)]
        )


class DynaPipeMicrobatch:
    # This class is used to represent a microbatch for DynaPipe, which can be
    # converted to/from a model spec json file. It is used to supply
    # arguments to the micro-batch generator and scheduler.
    def __init__(self, name) -> None:
        self.name = name
        self.n_layers = None
        # 执行时间和存储激活需要三种重计算方案
        self.fw_exec_times = [[], [], []]  # [none_times, full_times, selective_times]
        self.bw_exec_times = [[], [], []]
        self.model_stored_activation_memory = [[], [], []]
        # 这些不需要区分重计算方案
        self.fw_comm_size = []
        self.bw_comm_size = []
        self.model_state_memory = []
        self.model_peak_activation_memory = []  # 保持单列表
        self.activation_shapes = []

    def _check_or_set_nlayers(self, n_layers, debug_name, minus_one=False):
        expected_value = self.n_layers if not minus_one else self.n_layers - 1
        if self.n_layers is not None:
            assert (
                n_layers == expected_value
            ), """{} must have length n_layers {} ({}),
                but got length {}""".format(
                debug_name,
                "- 1" if minus_one else "",
                expected_value,
                n_layers,
            )
        else:
            self.n_layers = n_layers

    def set_fw_exec_times(self, fw_exec_times_list: List[List[float]]) -> None:
        assert len(fw_exec_times_list) == 3, "Must provide times for all 3 recompute types"
        n_layers = len(fw_exec_times_list[0])
        for times in fw_exec_times_list[1:]:
            assert len(times) == n_layers, "Inconsistent layer numbers across recompute types"
        self._check_or_set_nlayers(n_layers, "fw_exec_times")
        self.fw_exec_times = fw_exec_times_list

    def set_bw_exec_times(self, bw_exec_times_list: List[List[float]]) -> None:
        assert len(bw_exec_times_list) == 3, "Must provide times for all 3 recompute types"
        n_layers = len(bw_exec_times_list[0])
        for times in bw_exec_times_list[1:]:
            assert len(times) == n_layers, "Inconsistent layer numbers across recompute types"
        self._check_or_set_nlayers(n_layers, "bw_exec_times")
        self.bw_exec_times = bw_exec_times_list

    def set_fw_comm_size(self, fw_comm_size: List[float]) -> None:
        # size is in mega bytes (MB)
        self._check_or_set_nlayers(
            len(fw_comm_size), "fw_comm_size", minus_one=True
        )
        self.fw_comm_size = fw_comm_size

    def set_bw_comm_size(self, bw_comm_size: List[float]) -> None:
        # size is in mega bytes (MB)
        self._check_or_set_nlayers(
            len(bw_comm_size), "bw_comm_size", minus_one=True
        )
        self.bw_comm_size = bw_comm_size

    def set_model_state_memory(self, model_state_memory: List[float]) -> None:
        # size is in MB (megabytes)
        self._check_or_set_nlayers(
            len(model_state_memory), "model_state_memory"
        )
        self.model_state_memory = model_state_memory
        
    def set_model_stored_activation_memory(
        self, stored_activation_list: List[List[float]]
    ) -> None:
        assert len(stored_activation_list) == 3, "Must provide memory for all 3 recompute types"
        n_layers = len(stored_activation_list[0])
        for mem in stored_activation_list[1:]:
            assert len(mem) == n_layers, "Inconsistent layer numbers across recompute types"
        self._check_or_set_nlayers(n_layers, "model_stored_activation_memory")
        self.model_stored_activation_memory = stored_activation_list

    def set_model_peak_activation_memory(
        self, model_peak_activation_memory: List[float]
    ) -> None:
        # size is in MB (megabytes)
        self._check_or_set_nlayers(
            len(model_peak_activation_memory), "model_peak_activation_memory"
        )
        self.model_peak_activation_memory = model_peak_activation_memory

    def set_activation_shapes(
        self, activation_shapes: List[List[Tuple[int, int, int]]]
    ) -> None:
        # activation_shapes: outer list: layer, inner list: output activations
        # Note that for decoders, the activation should be the
        # output of encoder + decoder, since encoder output is needed for
        # all decoder layers.
        self._check_or_set_nlayers(len(activation_shapes), "activation_shapes")
        # make shapes immutable
        activation_shapes = [tuple(x) for x in activation_shapes]
        self.activation_shapes = activation_shapes

    def check_all_set(self):
        assert self.n_layers is not None
        assert len(self.fw_exec_times[0]) == self.n_layers
        assert len(self.bw_exec_times[0]) == self.n_layers
        assert len(self.fw_comm_size) == self.n_layers - 1
        assert len(self.bw_comm_size) == self.n_layers - 1
        assert len(self.model_state_memory) == self.n_layers
        assert len(self.model_stored_activation_memory[0]) == self.n_layers
        assert len(self.model_peak_activation_memory) == self.n_layers
        assert len(self.activation_shapes) == self.n_layers

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "n_layers": self.n_layers,
            "fw_exec_times": self.fw_exec_times,
            "bw_exec_times": self.bw_exec_times,
            "fw_comm_size": self.fw_comm_size,
            "bw_comm_size": self.bw_comm_size,
            "model_state_memory": self.model_state_memory,
            "model_stored_activation_memory": self.model_stored_activation_memory,  # noqa: E501
            "model_peak_activation_memory": self.model_peak_activation_memory,
            "activation_shapes": self.activation_shapes,
        }

    @staticmethod
    def from_json(json_dict):
        microbatch = DynaPipeMicrobatch(json_dict["name"])
        microbatch.set_fw_exec_times(json_dict["fw_exec_times"])
        microbatch.set_bw_exec_times(json_dict["bw_exec_times"])
        microbatch.set_fw_comm_size(json_dict["fw_comm_size"])
        microbatch.set_bw_comm_size(json_dict["bw_comm_size"])
        microbatch.set_model_state_memory(json_dict["model_state_memory"])
        microbatch.set_model_stored_activation_memory(
            json_dict["model_stored_activation_memory"]
        )
        microbatch.set_model_peak_activation_memory(
            json_dict["model_peak_activation_memory"]
        )
        microbatch.set_activation_shapes(json_dict["activation_shapes"])
        return microbatch


class DynaPipeMinibatch:
    # This class represents a list of microbatches (a minibatch)
    def __init__(
        self, name: str, microbatches: List[DynaPipeMicrobatch] = None
    ) -> None:
        self.name = name
        self.microbatches = microbatches if microbatches else []
        self.n_layers = None if not microbatches else microbatches[0].n_layers

    def add_microbatch(self, microbatch: DynaPipeMicrobatch) -> None:
        if self.n_layers is None:
            self.n_layers = microbatch.n_layers
        else:
            assert (
                self.n_layers == microbatch.n_layers
            ), "All microbatches must have the same number of layers"
        self.microbatches.append(microbatch)

    def __str__(self):
        return (
            "("
            + self.name
            + ", "
            + str(len(self.microbatches))
            + " microbatches)"
        )

    @staticmethod
    def from_json(json_dict):
        minibatch = DynaPipeMinibatch(json_dict["name"])
        json_list = json_dict["microbatches"]
        for json_dict in json_list:
            microbatch = DynaPipeMicrobatch.from_json(json_dict)
            minibatch.add_microbatch(microbatch)
        return minibatch

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "microbatches": [
                microbatch.to_json() for microbatch in self.microbatches
            ],
        }

    def permute_microbatches(self, permutation: List[int]) -> None:
        assert len(permutation) == len(self.microbatches)
        permuted_microbatches = [self.microbatches[i] for i in permutation]
        return DynaPipeMinibatch(self.name, permuted_microbatches)


class DynaPipeCluster:
    def __init__(
        self,
        device2node: Dict[int, int],
        memory_limits: List[int],
        intra_node_bw_gbps: float,
        inter_node_bw_gbps: float,
        intra_node_lat_us: float,
        inter_node_lat_us: float,
    ) -> None:
        # memory_limits is in MB (megabytes)
        # bw is in Gbps (gigabits per second)
        # lat is in us (microseconds)
        devices = set()
        nodes = set()
        for device, node in device2node.items():
            devices.add(device)
            nodes.add(node)
        self.n_devices = len(devices)
        self.n_nodes = len(nodes)
        self.device2node = device2node
        flattened_devices = [device for device in device2node.keys()]
        assert list(sorted(list(set(flattened_devices)))) == list(
            range(self.n_devices)
        ), "Device ids must be contiguous and start at 0"
        assert len(memory_limits) == self.n_devices, (
            "Expected memory limits for each of the "
            f"{self.n_devices} devices, but got "
            f"{len(memory_limits)} numbers."
        )
        self.memory_limits = memory_limits
        self.intra_node_bw = intra_node_bw_gbps
        self.inter_node_bw = inter_node_bw_gbps
        self.intra_node_lat = intra_node_lat_us
        self.inter_node_lat = inter_node_lat_us

    def _get_bw(self, dev0, dev1):
        if self.device2node[dev0] == self.device2node[dev1]:
            return self.intra_node_bw
        else:
            return self.inter_node_bw

    def _get_lat(self, dev0, dev1):
        if self.device2node[dev0] == self.device2node[dev1]:
            return self.intra_node_lat
        else:
            return self.inter_node_lat

    def get_comm_time(self, megabytes, dev0, dev1):
        if dev0 == dev1:
            return 0
        return self._get_lat(dev0, dev1) + 1e6 * (
            megabytes * 8 / 1e3
        ) / self._get_bw(dev0, dev1)

    def get_memory_limit(self, dev):
        return self.memory_limits[dev]

    def to_json(self) -> dict:
        return {
            "n_devices": self.n_devices,
            "n_nodes": self.n_nodes,
            "device2node": self.device2node,
            "memory_limits": self.memory_limits,
            "intra_node_bw": self.intra_node_bw,
            "inter_node_bw": self.inter_node_bw,
            "intra_node_lat": self.intra_node_lat,
            "inter_node_lat": self.inter_node_lat,
        }

    def dumps(self) -> str:
        return json.dumps(self.to_json())

    @staticmethod
    def loads(json_str: str) -> "DynaPipeCluster":
        return DynaPipeCluster.from_json(json.loads(json_str))

    @staticmethod
    def from_json(json_dict):
        converted_device2node = {
            int(k): int(v) for k, v in json_dict["device2node"].items()
        }
        json_dict["device2node"] = converted_device2node
        cluster = DynaPipeCluster(
            json_dict["device2node"],
            json_dict["memory_limits"],
            json_dict["intra_node_bw"],
            json_dict["inter_node_bw"],
            json_dict["intra_node_lat"],
            json_dict["inter_node_lat"],
        )
        return cluster


def get_uniform_microbatch(
    n_layers,
    comm_ratio=1,
    bw_multiplier=1.0,
    comp_time=1000,
    comm_size=12.5,
    per_layer_state_memory=0,
    per_layer_stored_activation_memory=1000,
    per_layer_peak_activation_memory=2000,
):
    comm_size = comm_size * comm_ratio
    microbatch = DynaPipeMicrobatch("uniform")
    microbatch.set_fw_exec_times([[comp_time] * n_layers] * 3)
    microbatch.set_bw_exec_times([[comp_time * bw_multiplier] * n_layers] * 3)
    microbatch.set_fw_comm_size([comm_size] * (n_layers - 1))
    microbatch.set_bw_comm_size([comm_size * bw_multiplier] * (n_layers - 1))
    microbatch.set_model_state_memory([per_layer_state_memory] * n_layers)
    microbatch.set_model_stored_activation_memory(
        [[per_layer_stored_activation_memory] * n_layers] * 3
    )
    microbatch.set_model_peak_activation_memory(
        [per_layer_peak_activation_memory] * n_layers
    )
    microbatch.set_activation_shapes([[(32, 128, 128)]] * n_layers)
    return microbatch


def get_uniform_cluster(n_devices, intra_node_bw=4800, inter_node_bw=100):
    device2node = {i: i for i in range(n_devices)}
    memory_limits = [1000000] * n_devices
    cluster = DynaPipeCluster(
        device2node, memory_limits, intra_node_bw, inter_node_bw, 0, 0
    )
    return cluster


def get_example_ende_microbatch(
    n_layers_en,
    n_layers_de,
    comm_ratio=1,
    bw_multiplier=1.0,
    en_comp_time=1000,
    en_comm_size=12500000,
    de_ratio=0.5,
    en_per_layer_state_memory=0,
    per_layer_stored_activation_memory=1000,
    per_layer_peak_activation_memory=2000,
):
    en_comm_size = en_comm_size * comm_ratio
    microbatch = DynaPipeMicrobatch("ende")
    microbatch.set_fw_exec_times(
        [[en_comp_time] * n_layers_en + [en_comp_time * de_ratio] * n_layers_de] * 3
    )
    microbatch.set_bw_exec_times(
        [[bw_multiplier * x for x in list(reversed(microbatch.fw_exec_times[0]))]] * 3
    )
    microbatch.set_fw_comm_size(
        [en_comm_size] * n_layers_en + [en_comm_size * 2] * (n_layers_de - 1)
    )
    microbatch.set_bw_comm_size(list(reversed(microbatch.fw_comm_size)))
    microbatch.set_model_state_memory(
        [en_per_layer_state_memory] * n_layers_en
        + [en_per_layer_state_memory * de_ratio] * n_layers_de
    )
    microbatch.set_model_stored_activation_memory(
        [[per_layer_stored_activation_memory] * n_layers_en + [per_layer_stored_activation_memory * de_ratio] * n_layers_de] * 3
    )
    microbatch.set_model_peak_activation_memory(
        [per_layer_peak_activation_memory] * n_layers_en
        + [per_layer_peak_activation_memory * de_ratio] * n_layers_de
    )
    microbatch.set_activation_shapes(
        [[(32, 512, 512)]] * n_layers_en
        + [[(32, 128, 512), (32, 128, 512)]] * n_layers_de
    )
    return microbatch


def get_device_memory_limits(dpp_cluster: DynaPipeCluster):
    return [
        dpp_cluster.get_memory_limit(i) for i in range(dpp_cluster.n_devices)
    ]


def get_simulator(
    scheduler_type: str,
    dpp_minibatch: DynaPipeMinibatch,
    dpp_cluster: DynaPipeCluster,
    device_assignment: List[int],
    rc_plan: Optional[List[List[Tuple[int, int, int]]]] = None,
    include_memory_stats: bool = True,
    memory_limit: float = float("inf"),
    max_otf_microbatches: Union[None, int] = None,
    logger=None,
) -> Scheduler:
    assert len(device_assignment) == dpp_minibatch.n_layers, (
        "Device assignment must be specified for each layer. "
        "Expected {} layers, but got {}.".format(
            dpp_minibatch.n_layers, len(device_assignment)
        )
    )
    bw_device_assignment = list(reversed(device_assignment))
    microbatch_specs = []
    for mb_idx, microbatch in enumerate(dpp_minibatch.microbatches):
        if rc_plan is not None:
            mb_rc_plan = rc_plan[mb_idx]
            fw_times = []
            bw_times = []
            stored_activation = []
            
            for layer_idx in range(microbatch.n_layers):
                rc_type = 0  # 默认使用 none
                for start, end, rc in mb_rc_plan:
                    if start <= layer_idx < end:
                        rc_type = rc
                        break
                        
                fw_times.append(microbatch.fw_exec_times[rc_type][layer_idx])
                bw_times.append(microbatch.bw_exec_times[rc_type][layer_idx])
                stored_activation.append(microbatch.model_stored_activation_memory[rc_type][layer_idx])
        else:
            raise NotImplementedError("no rc_plan is not supported for now")
            # 默认使用 none (rc_type=0)
            fw_times = microbatch.fw_exec_times[0]
            bw_times = microbatch.bw_exec_times[0]
            stored_activation = microbatch.model_stored_activation_memory[0]
            mb_rc_plan = None

        fw_comm_times = [
            dpp_cluster.get_comm_time(
                microbatch.fw_comm_size[i],
                device_assignment[i],
                device_assignment[i + 1],
            )
            for i in range(microbatch.n_layers - 1)
        ]
        bw_comm_times = [
            dpp_cluster.get_comm_time(
                microbatch.bw_comm_size[i],
                bw_device_assignment[i],
                bw_device_assignment[i + 1],
            )
            for i in range(microbatch.n_layers - 1)
        ]
        fw_model_state = microbatch.model_state_memory
        fw_peak_activation = microbatch.model_peak_activation_memory
        microbatch_spec = SchedulerMicrobatchSpec(
            name=microbatch.name,
            fw_times=fw_times,
            fw_comm_times=fw_comm_times,
            fw_stored_activation_size=stored_activation,
            fw_peak_activation_size=fw_peak_activation,
            bw_times=bw_times,
            bw_comm_times=bw_comm_times,
            activation_shapes=microbatch.activation_shapes,
            rc_plan=mb_rc_plan,
        )
        microbatch_specs.append(microbatch_spec)
    minibatch_spec = SchedulerMinibatchSpec(
        dpp_minibatch.name,
        microbatch_specs,
        device_assignment,
        fw_model_state,
    )
    scheduler_class = get_scheduler_class(scheduler_type)
    if max_otf_microbatches is not None:
        assert scheduler_type in ["cyclic", "wait-free-cyclic"], (
            "max_otf_microbatches is only supported for cyclic "
            "and wait-free-cyclic schedulers. "
            "Got scheduler_type={}".format(scheduler_type)
        )
        simulator = scheduler_class(
            minibatch_spec,
            include_memory_stats=include_memory_stats,
            memory_limit=memory_limit,
            max_otf_microbatches=max_otf_microbatches,
            logger=logger,
        )
    else:
        simulator = scheduler_class(
            minibatch_spec,
            include_memory_stats=include_memory_stats,
            memory_limit=memory_limit,
            logger=logger,
        )
    return simulator
