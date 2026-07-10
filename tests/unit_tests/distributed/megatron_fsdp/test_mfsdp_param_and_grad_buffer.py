# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as pgb_module
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    Bucket,
    BucketingPolicy,
    GradReducePipeline,
    ParamAndGradBuffer,
    PlannedUnitDoubleBufferAllocator,
    TemporaryBucketAllocator,
    _get_parameter_groups,
)


class _ExpertTestModule(torch.nn.Module):
    """
    Mock module whose params are routed under `.experts.` to trigger
    is_expert_param=True. The outer `layer` attribute puts a dot before
    `experts` in the parameter path (e.g. `layer.experts.linear_fc1`).
    """

    def __init__(self, shapes):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.experts = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.empty(shape)) for name, shape in shapes.items()}
        )


def _get_bucket_signatures(module):
    bucket_groups, _, _ = _get_parameter_groups(
        module, BucketingPolicy(suggested_bucket_size=None), meta_device_init_fp8_params={}
    )
    param_to_name = {param: name for name, param in module.named_parameters()}
    return [
        {
            "chunk_size_factor": group.chunk_size_factor,
            "params": [(param_to_name[param], tuple(param.shape)) for param in group.params],
        }
        for group in bucket_groups
    ]


def test_grouped_expert_weights_split_when_chunk_size_factors_differ():
    """Grouped expert weights with mismatched chunk size factors get routed to separate buckets."""
    num_local_experts = 4
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (num_local_experts, 2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (num_local_experts, hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc1"])[1:].numel(),
            "params": [("layer.experts.linear_fc1", shapes["linear_fc1"])],
        },
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc2"])[1:].numel(),
            "params": [("layer.experts.linear_fc2", shapes["linear_fc2"])],
        },
    ]


def test_per_expert_2d_weights_merge_via_lcm():
    """Per-expert 2D weights merge into a single bucket via LCM chunk size factor."""
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": math.lcm(
                torch.Size(shapes["linear_fc1"])[1:].numel(),
                torch.Size(shapes["linear_fc2"])[1:].numel(),
            ),
            "params": [
                ("layer.experts.linear_fc1", shapes["linear_fc1"]),
                ("layer.experts.linear_fc2", shapes["linear_fc2"]),
            ],
        }
    ]


class _CpuGlobalMemoryBuffer:
    """CPU test double with GlobalMemoryBuffer's grow-on-demand contract."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        required_len = math.prod(tensor_shape)
        key = (name, dtype)
        if key not in self.buffer or self.buffer[key].numel() < required_len:
            allocation_context = mem_alloc_context or nullcontext
            with allocation_context():
                self.buffer[key] = torch.empty(required_len, dtype=dtype)
        return self.buffer[key][:required_len].view(*tensor_shape)


class _TrackingBucketAllocator(TemporaryBucketAllocator):
    """CPU allocator that exposes whether a bucket used dynamic storage."""

    def __init__(self):
        super().__init__()
        self.allocate_calls = []
        self.free_calls = []

    def allocate(self, bucket_id, size, dtype, device, mem_alloc_context=None):
        self.allocate_calls.append((bucket_id, size, dtype, device))
        self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        return self.buckets[bucket_id]

    def free(self, bucket_id):
        self.free_calls.append(bucket_id)
        self.buckets.pop(bucket_id, None)


def _allocator_param_groups(*unit_ids):
    return [
        SimpleNamespace(
            fsdp_unit_id=unit_id,
            dtype=torch.float32,
            params=[torch.nn.Parameter(torch.empty(2))],
        )
        for unit_id in unit_ids
    ]


def _allocate_and_free_unit(allocator, bucket_sizes):
    allocated = []
    for bucket_id, size, dtype in bucket_sizes:
        allocated.append(
            allocator.allocate(bucket_id, size, dtype, torch.device("cpu"))
        )
    for bucket_id, _, _ in bucket_sizes:
        allocator.free(bucket_id)
    return allocated


@pytest.fixture
def cpu_global_memory_buffer(monkeypatch):
    global_memory_buffer = _CpuGlobalMemoryBuffer()
    monkeypatch.setattr(pgb_module, "get_global_memory_buffer", lambda: global_memory_buffer)
    monkeypatch.setattr(pgb_module.torch.distributed, "is_initialized", lambda: False)
    return global_memory_buffer


def test_planned_unit_double_buffer_uses_unit_colors_and_max_unit_arena(
    cpu_global_memory_buffer,
):
    """All decoder units are planned in banks sized to the largest real unit."""
    # Units 0 and 1 each have two buckets. Unit 2 is configured but is not in
    # graph_bucket_ids, exercising eager decoder layers that still share the DB.
    param_groups = _allocator_param_groups(0, 0, 1, 1, 2, 2)
    base = _TrackingBucketAllocator()
    allocator = PlannedUnitDoubleBufferAllocator("decoder", base, param_groups)
    configured_meta = {
        0: (8, torch.float32),
        1: (3, torch.float32),
        2: (5, torch.float32),
        3: (10, torch.float32),
        4: (6, torch.float32),
        5: (4, torch.float32),
    }
    allocator.configure_units(configured_meta)

    _allocate_and_free_unit(
        allocator, [(0, 8, torch.float32), (1, 3, torch.float32)]
    )
    _allocate_and_free_unit(
        allocator, [(2, 5, torch.float32), (3, 10, torch.float32)]
    )
    _allocate_and_free_unit(
        allocator, [(4, 6, torch.float32), (5, 4, torch.float32)]
    )
    allocator.freeze_plan({0, 1})

    assert allocator.configured_units == {0, 1, 2}
    assert set(allocator.unit_plan) == {0, 1, 2}
    assert allocator.unit_plan[0] == allocator.unit_plan[1] == allocator.unit_plan[2] == 0
    assert allocator._arena_capacities == {torch.float32: 15}
    assert allocator._bucket_layout == {
        (0, torch.float32): (0, 8),
        (1, torch.float32): (8, 3),
        (2, torch.float32): (0, 5),
        (3, torch.float32): (5, 10),
        (4, torch.float32): (0, 6),
        (5, torch.float32): (6, 4),
    }

    # Each unit's buckets occupy non-overlapping slices. Different units may
    # reuse those offsets because the unit-level color prevents overlap.
    for first_bucket, second_bucket in ((0, 1), (2, 3), (4, 5)):
        first_offset, first_size = allocator._bucket_layout[(first_bucket, torch.float32)]
        second_offset, second_size = allocator._bucket_layout[(second_bucket, torch.float32)]
        assert first_offset + first_size <= second_offset
        assert second_offset + second_size <= allocator._arena_capacities[torch.float32]

    float32_bytes = torch.empty((), dtype=torch.float32).element_size()
    max_unit_elements = max(8 + 3, 5 + 10, 6 + 4)
    assert allocator._arena_capacities[torch.float32] == max_unit_elements
    assert allocator.materialized_bytes == allocator.NUM_BANKS * max_unit_elements * float32_bytes
    assert len(allocator._slot_materialization) == allocator.NUM_BANKS

    first = allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    second = allocator.allocate(1, 3, torch.float32, torch.device("cpu"))
    assert second.data.data_ptr() == first.data.data_ptr() + 8 * float32_bytes
    allocator.free(0)
    allocator.free(1)
    eager_only = allocator.allocate(4, 6, torch.float32, torch.device("cpu"))
    assert eager_only.data.data_ptr() == first.data.data_ptr()
    allocator.free(4)

    allocator.freeze_plan({0, 1})
    with pytest.raises(RuntimeError, match="new graph buckets appeared"):
        allocator.freeze_plan({0, 1, 2})


def test_planned_unit_double_buffer_keeps_bank_until_every_unit_lane_is_freed(
    cpu_global_memory_buffer,
):
    """Overlapping units use two banks and a partial free cannot release a bank."""
    param_groups = _allocator_param_groups(0, 0, 1, 1, 2, 2)
    allocator = PlannedUnitDoubleBufferAllocator(
        "overlap", _TrackingBucketAllocator(), param_groups
    )
    allocator.configure_units(
        {bucket_id: (8, torch.float32) for bucket_id in range(len(param_groups))}
    )

    # Units 0 and 1 overlap. Unit 2 is sequential with unit 0 and therefore
    # reuses unit 0's bank after freeze.
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.allocate(1, 8, torch.float32, torch.device("cpu"))
    allocator.allocate(2, 8, torch.float32, torch.device("cpu"))
    allocator.allocate(3, 8, torch.float32, torch.device("cpu"))
    allocator.free(2)
    allocator.free(3)
    allocator.free(0)
    allocator.free(1)
    _allocate_and_free_unit(
        allocator, [(4, 8, torch.float32), (5, 8, torch.float32)]
    )
    allocator.freeze_plan(range(6))

    assert allocator.unit_plan[0] != allocator.unit_plan[1]
    assert allocator.unit_plan[0] == allocator.unit_plan[2]

    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.allocate(1, 8, torch.float32, torch.device("cpu"))
    allocator.allocate(2, 8, torch.float32, torch.device("cpu"))
    allocator.allocate(3, 8, torch.float32, torch.device("cpu"))

    allocator.free(0)
    assert allocator._bank_using[allocator.unit_plan[0]] == 0
    with pytest.raises(RuntimeError, match="planned bank.*held by FSDP unit 0"):
        allocator.allocate(4, 8, torch.float32, torch.device("cpu"))

    allocator.free(1)
    allocator.allocate(4, 8, torch.float32, torch.device("cpu"))
    allocator.free(4)
    allocator.free(2)
    allocator.free(3)


def test_planned_unit_double_buffer_colors_throttled_claim_trace_with_two_banks(
    cpu_global_memory_buffer,
):
    """Freeing A before claiming C keeps the warmup conflict graph bipartite."""
    param_groups = _allocator_param_groups(0, 1, 2)
    allocator = PlannedUnitDoubleBufferAllocator(
        "throttled_claim", _TrackingBucketAllocator(), param_groups
    )
    allocator.configure_units(
        {bucket_id: (8, torch.float32) for bucket_id in range(len(param_groups))}
    )

    allocator.record_graph_bucket_claim(0, 8, torch.float32)
    allocator.record_graph_bucket_claim(1, 8, torch.float32)
    allocator.free(0)
    allocator.record_graph_bucket_claim(2, 8, torch.float32)
    allocator.free(1)
    allocator.free(2)
    allocator.freeze_plan(range(3))

    assert allocator.unit_plan[0] == allocator.unit_plan[2]
    assert allocator.unit_plan[0] != allocator.unit_plan[1]


def test_planned_unit_double_buffer_rejects_a_third_live_unit(cpu_global_memory_buffer):
    """A warmup topology requiring three simultaneous banks fails at freeze."""
    param_groups = _allocator_param_groups(0, 1, 2)
    allocator = PlannedUnitDoubleBufferAllocator(
        "three_live", _TrackingBucketAllocator(), param_groups
    )
    allocator.configure_units(
        {bucket_id: (8, torch.float32) for bucket_id in range(len(param_groups))}
    )

    for bucket_id in range(3):
        allocator.allocate(bucket_id, 8, torch.float32, torch.device("cpu"))
    for bucket_id in reversed(range(3)):
        allocator.free(bucket_id)

    with pytest.raises(RuntimeError, match="more than two.*unit banks"):
        allocator.freeze_plan(range(3))


def test_planned_unit_double_buffer_leaves_unconfigured_vision_unit_dynamic(
    cpu_global_memory_buffer,
):
    """A Vision FSDP unit never enters the decoder plan or a fixed bank."""
    param_groups = _allocator_param_groups(0, 99)
    base = _TrackingBucketAllocator()
    allocator = PlannedUnitDoubleBufferAllocator("vision_dynamic", base, param_groups)
    allocator.configure_units({0: (8, torch.float32)})

    _allocate_and_free_unit(allocator, [(0, 8, torch.float32)])
    _allocate_and_free_unit(allocator, [(1, 7, torch.float32)])
    allocator.freeze_plan({0})

    dynamic = allocator.allocate(1, 7, torch.float32, torch.device("cpu"))
    assert allocator.configured_units == {0}
    assert 99 not in allocator.unit_plan
    assert base.buckets[1] is dynamic
    assert base.allocate_calls[-1][:3] == (1, 7, torch.float32)
    assert allocator._bank_using == {}
    allocator.free(1)
    assert base.free_calls[-1] == 1


def test_planned_unit_double_buffer_claim_is_idempotent_and_released_for_reuse(
    cpu_global_memory_buffer,
):
    """Pre-write claims share unit occupancy and reuse the normal free lifecycle."""
    param_groups = _allocator_param_groups(0, 1)
    allocator = PlannedUnitDoubleBufferAllocator(
        "grad_claim", _TrackingBucketAllocator(), param_groups
    )
    allocator.configure_units({0: (8, torch.float32), 1: (8, torch.float32)})
    _allocate_and_free_unit(allocator, [(0, 8, torch.float32)])
    _allocate_and_free_unit(allocator, [(1, 8, torch.float32)])
    allocator.freeze_plan({0, 1})
    assert allocator.unit_plan[0] == allocator.unit_plan[1]

    allocator.claim_graph_bucket(0, 8, torch.float32)
    allocator.claim_graph_bucket(0, 8, torch.float32)
    assert allocator._unit_live_buckets[0] == {0}
    with pytest.raises(RuntimeError, match="planned bank.*held by FSDP unit 0"):
        allocator.claim_graph_bucket(1, 8, torch.float32)

    allocator.free(0)
    allocator.claim_graph_bucket(1, 8, torch.float32)
    assert allocator._bank_using[allocator.unit_plan[1]] == 1
    allocator.free(1)
    assert allocator._bank_using == {}

    with pytest.raises(RuntimeError, match="reserved 8 elements.*requests 9"):
        allocator.claim_graph_bucket(0, 9, torch.float32)
    with pytest.raises(RuntimeError, match="reserved 8 elements.*requests 7"):
        allocator.claim_graph_bucket(0, 7, torch.float32)
    with pytest.raises(RuntimeError, match="reserved 8 elements.*requests 7"):
        allocator.allocate(0, 7, torch.float32, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="has no frozen.*arena slice"):
        allocator.claim_graph_bucket(0, 8, torch.float64)


def test_planned_grad_arena_claim_detects_comm_dtype_pointer_drift_before_write(
    cpu_global_memory_buffer,
):
    """A fused-wgrad claim validates the communication arena used by graph replay."""
    allocator = PlannedUnitDoubleBufferAllocator(
        "pointer_drift", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    allocator.configure_units({0: (8, torch.float32)})
    allocator.record_graph_bucket_claim(0, 8, torch.bfloat16)
    _allocate_and_free_unit(allocator, [(0, 8, torch.bfloat16)])
    allocator.freeze_plan({0})

    bank = allocator.unit_plan[0]
    slot = (bank, torch.bfloat16)
    key = (allocator._buffer_name(bank), torch.bfloat16)
    original_tensor = cpu_global_memory_buffer.buffer[key]
    cpu_global_memory_buffer.buffer[key] = torch.empty(
        original_tensor.numel() + 1, dtype=torch.bfloat16
    )

    with pytest.raises(RuntimeError, match="moved from address"):
        allocator.claim_graph_bucket(0, 8, torch.bfloat16)

    assert allocator._bank_using == {}
    assert allocator._slot_materialization[slot][2] == original_tensor.data_ptr()
    assert original_tensor.data_ptr() != cpu_global_memory_buffer.buffer[key].data_ptr()


def test_fused_wgrad_claim_records_throttles_and_claims_in_order():
    """Fused replay claims are deduplicated, while eager buckets are record-only."""
    fused_param = torch.nn.Parameter(torch.empty(1))
    non_fused_param = torch.nn.Parameter(torch.empty(1))
    events = []

    main_grad_allocator = SimpleNamespace(
        record_graph_bucket_claim=lambda bucket_id, size, dtype: events.append(
            ("record", bucket_id, size, dtype)
        ),
        claim_graph_bucket=lambda bucket_id, size, dtype: events.append(
            ("claim", bucket_id, size, dtype)
        ),
    )
    fused_grad_buffer = SimpleNamespace(
        is_data_distributed=True,
        temporary_bucket_allocator=main_grad_allocator,
        bucket_index=SimpleNamespace(size=8),
        dtype=torch.float32,
    )
    non_fused_grad_buffer = SimpleNamespace(
        is_data_distributed=True,
        temporary_bucket_allocator=main_grad_allocator,
        bucket_index=SimpleNamespace(size=12),
        dtype=torch.float32,
    )
    pgb = SimpleNamespace(
        main_grad_alloc=main_grad_allocator,
        mp_policy=SimpleNamespace(grad_comm_dtype=torch.bfloat16),
        param_to_param_group={fused_param: 0, non_fused_param: 1},
        parameter_groups=[
            SimpleNamespace(hfsdp_helper_gbuf=None, main_grad_buffer=fused_grad_buffer),
            SimpleNamespace(hfsdp_helper_gbuf=None, main_grad_buffer=non_fused_grad_buffer),
        ],
    )
    fake_fsdp = SimpleNamespace(
        param_and_grad_buffer=pgb,
        _cuda_graph_fused_wgrad_params={id(fused_param)},
        grad_reduce_pipeline=SimpleNamespace(
            _enforce_double_buffer_limit=lambda bucket_ids: events.append(
                ("throttle", tuple(bucket_ids))
            )
        ),
    )

    MegatronFSDP._claim_cuda_graph_fused_wgrad_buckets(
        fake_fsdp,
        [fused_param, non_fused_param, fused_param, non_fused_param],
    )

    assert events == [
        ("throttle", (0, 1)),
        ("record", 0, 8, torch.bfloat16),
        ("record", 1, 12, torch.bfloat16),
        ("claim", 0, 8, torch.bfloat16),
    ]


@pytest.mark.parametrize(
    ("queued_bucket_ids", "expected_keep_n", "expected_remaining"),
    [
        pytest.param([0, 1, 2], 1, [2], id="vision-before-two-decoder-units"),
        pytest.param([1, 0, 2], 2, [0, 2], id="vision-between-decoder-units"),
    ],
)
def test_grad_reduce_double_buffer_cutoff_counts_vision_queue_positions(
    monkeypatch,
    queued_bucket_ids,
    expected_keep_n,
    expected_remaining,
):
    """Vision uses no bank, but its queue position remains part of the wait cutoff."""
    vision_unit = 99
    decoder_unit_a = 10
    decoder_unit_b = 11
    incoming_unit_c = 12
    buffer = SimpleNamespace(
        ddp_config=SimpleNamespace(fsdp_double_buffer=True),
        parameter_groups=[
            SimpleNamespace(fsdp_unit_id=vision_unit),
            SimpleNamespace(fsdp_unit_id=decoder_unit_a),
            SimpleNamespace(fsdp_unit_id=decoder_unit_b),
            SimpleNamespace(fsdp_unit_id=incoming_unit_c),
        ],
        double_buf_units={decoder_unit_a, decoder_unit_b, incoming_unit_c},
    )
    pipeline = GradReducePipeline.__new__(GradReducePipeline)
    pipeline.buffer = buffer
    pipeline.rs_stream = object()
    pipeline.grad_reduce_queue = [
        (None, None, bucket_id) for bucket_id in queued_bucket_ids
    ]
    wait_calls = []
    consumer_stream = object()

    def record_wait(keep_n, wait_stream=None):
        wait_calls.append((keep_n, wait_stream))
        if keep_n:
            pipeline.grad_reduce_queue[:] = pipeline.grad_reduce_queue[-keep_n:]
        else:
            pipeline.grad_reduce_queue.clear()

    pipeline.wait_for_previous_grad_reduce = record_wait
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: consumer_stream)

    pipeline._enforce_double_buffer_limit([3])

    assert wait_calls == [(expected_keep_n, consumer_stream)]
    assert [
        bucket_id for _, _, bucket_id in pipeline.grad_reduce_queue
    ] == expected_remaining


def test_grad_reduce_double_buffer_orders_consumer_stream_before_bank_release(monkeypatch):
    """A reused bank must not race with a prior reduce-scatter reader."""
    consumer_stream = object()
    events = []
    buffer = SimpleNamespace(
        ddp_config=SimpleNamespace(fsdp_double_buffer=True),
        parameter_groups=[
            SimpleNamespace(fsdp_unit_id=10),
            SimpleNamespace(fsdp_unit_id=11),
            SimpleNamespace(fsdp_unit_id=12),
        ],
        double_buf_units={10, 11, 12},
        num_buckets=3,
    )
    pipeline = GradReducePipeline.__new__(GradReducePipeline)
    pipeline.buffer = buffer
    pipeline.grad_reduce_queue = [
        (
            SimpleNamespace(
                wait=lambda stream=None: events.append(("wait", stream))
            ),
            lambda: events.append(("free", 0)),
            0,
        ),
        (None, None, 1),
    ]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: consumer_stream)

    pipeline._enforce_double_buffer_limit([2])

    assert events == [("wait", consumer_stream), ("free", 0)]
    assert [bucket_id for _, _, bucket_id in pipeline.grad_reduce_queue] == [1]


def test_pgb_freeze_rejects_graph_buffer_missing_from_configured_planned_buckets():
    """Intersecting with configured IDs must not hide a graph-covered planned buffer."""
    parameter_groups = _allocator_param_groups(0, 1)
    weight_allocator = PlannedUnitDoubleBufferAllocator(
        "weight", _TrackingBucketAllocator(), parameter_groups
    )
    transpose_allocator = PlannedUnitDoubleBufferAllocator(
        "transpose", _TrackingBucketAllocator(), parameter_groups
    )
    grad_allocator = PlannedUnitDoubleBufferAllocator(
        "grad", _TrackingBucketAllocator(), parameter_groups
    )
    weight_allocator.configure_units({0: (8, torch.float32)})

    parameter_groups[0].model_weight_buffer = None
    parameter_groups[0].transpose_weight_buffer = None
    parameter_groups[0].main_grad_buffer = None
    parameter_groups[1].model_weight_buffer = SimpleNamespace(
        is_data_distributed=True,
        temporary_bucket_allocator=weight_allocator,
    )
    parameter_groups[1].transpose_weight_buffer = None
    parameter_groups[1].main_grad_buffer = None
    fake_pgb = SimpleNamespace(
        ddp_config=SimpleNamespace(megatron_fsdp_use_planned_double_buffer=True),
        parameter_groups=parameter_groups,
        weight_alloc=weight_allocator,
        transpose_weight_alloc=transpose_allocator,
        main_grad_alloc=grad_allocator,
    )

    with pytest.raises(RuntimeError, match="not configured in a planned decoder FSDP unit"):
        ParamAndGradBuffer.freeze_planned_double_buffer(fake_pgb, {1})


def test_planned_allocators_keep_addresses_across_graph_eval_graph_lifecycle(
    cpu_global_memory_buffer,
):
    """Eval reuses decoder arenas, leaves Vision dynamic, and preserves graph addresses."""
    parameter_groups = _allocator_param_groups(0, 99)
    weight_base = _TrackingBucketAllocator()
    grad_base = _TrackingBucketAllocator()
    weight_allocator = PlannedUnitDoubleBufferAllocator(
        "lifecycle_weight", weight_base, parameter_groups
    )
    grad_allocator = PlannedUnitDoubleBufferAllocator(
        "lifecycle_grad", grad_base, parameter_groups
    )
    for allocator in (weight_allocator, grad_allocator):
        allocator.configure_units({0: (8, torch.float32)})

    # Eager warmup establishes the decoder unit lifetime before both plans freeze.
    _allocate_and_free_unit(weight_allocator, [(0, 8, torch.float32)])
    grad_allocator.record_graph_bucket_claim(0, 8, torch.float32)
    _allocate_and_free_unit(grad_allocator, [(0, 8, torch.float32)])
    weight_allocator.freeze_plan({0})
    grad_allocator.freeze_plan({0})

    def run_graph_train_stage():
        weight_bucket = weight_allocator.allocate(
            0, 8, torch.float32, torch.device("cpu")
        )
        weight_ptr = weight_bucket.data.data_ptr()
        weight_allocator.free(0)

        grad_allocator.claim_graph_bucket(0, 8, torch.float32)
        grad_bank = grad_allocator.unit_plan[0]
        grad_ptr = grad_allocator._slot_materialization[
            (grad_bank, torch.float32)
        ][2]
        grad_allocator.free(0)

        assert weight_allocator._bank_using == {}
        assert grad_allocator._bank_using == {}
        return weight_ptr, grad_ptr

    first_graph_weight_ptr, first_graph_grad_ptr = run_graph_train_stage()

    # Eval runs the decoder eagerly but still uses its frozen arena. Vision is
    # outside the configured decoder units and therefore stays on dynamic storage.
    eval_decoder_bucket = weight_allocator.allocate(
        0, 8, torch.float32, torch.device("cpu")
    )
    eval_decoder_ptr = eval_decoder_bucket.data.data_ptr()
    weight_allocator.free(0)
    vision_bucket = weight_allocator.allocate(
        1, 7, torch.float32, torch.device("cpu")
    )
    assert weight_base.buckets[1] is vision_bucket
    weight_allocator.free(1)

    assert weight_allocator._bank_using == {}
    assert grad_allocator._bank_using == {}
    assert [call[0] for call in weight_base.allocate_calls].count(1) == 1

    second_graph_weight_ptr, second_graph_grad_ptr = run_graph_train_stage()

    assert (
        first_graph_weight_ptr
        == eval_decoder_ptr
        == second_graph_weight_ptr
    )
    assert first_graph_grad_ptr == second_graph_grad_ptr
    assert weight_allocator._bank_using == {}
    assert grad_allocator._bank_using == {}
