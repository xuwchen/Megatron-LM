# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    AllGatherPipeline,
    BucketStatus,
)


class _FakeAllocator:
    def __init__(self):
        self.idle = ["slot0", "slot1"]
        self.using = {}

    def allocate(self, bucket_id):
        if bucket_id not in self.using:
            self.using[bucket_id] = self.idle.pop(0)
        return self.using[bucket_id]

    def free(self, bucket_id):
        if bucket_id not in self.using:
            return
        # Reuse recently freed slots first. This makes the address-reuse hazard
        # deterministic in the test.
        self.idle.insert(0, self.using.pop(bucket_id))


class _FakeWeightBuffer:
    def __init__(self, bucket_id=0, allocator=None):
        self.bucket_id = bucket_id
        self.allocator = allocator or _FakeAllocator()
        self.address = self.allocator.allocate(bucket_id)
        self.freed = False

    def free_bucket_storage(self):
        self.allocator.free(self.bucket_id)
        self.address = None
        self.freed = True


class _FakeParamGroup:
    def __init__(self, bucket_id=0, fsdp_unit_id=0, allocator=None):
        self.fsdp_unit_id = fsdp_unit_id
        self.model_weight_buffer = _FakeWeightBuffer(bucket_id=bucket_id, allocator=allocator)
        self.transpose_weight_buffer = None


def _make_pipeline(status=BucketStatus.READY_TO_USE):
    allocator = _FakeAllocator()
    param_group = _FakeParamGroup(allocator=allocator)
    pipeline = object.__new__(AllGatherPipeline)
    pipeline.buffer = SimpleNamespace(num_buckets=1, parameter_groups=[param_group])
    pipeline.param_gather_event_map = {}
    pipeline.bucket_status = {(0, False): status}
    pipeline.bucket_can_be_released = {(0, False): False}
    pipeline.cuda_graph_pinned_bucket_keys = set()
    return pipeline, param_group.model_weight_buffer


def test_cuda_graph_pinned_direct_release_preserves_storage():
    pipeline, weight_buffer = _make_pipeline()

    pipeline.pin_cuda_graph_bucket(0, bwd=False)
    pipeline.release_bucket(0, bwd=False)

    assert not weight_buffer.freed
    assert pipeline.bucket_status[(0, False)] == BucketStatus.PRESERVED
    assert pipeline.bucket_can_be_released[(0, False)] is False


def test_cuda_graph_pinned_lazy_recycle_preserves_storage():
    pipeline, weight_buffer = _make_pipeline()

    pipeline.pin_cuda_graph_bucket(0, bwd=False)
    pipeline.bucket_can_be_released[(0, False)] = True
    pipeline.recycle_unused_buckets()

    assert not weight_buffer.freed
    assert pipeline.bucket_status[(0, False)] == BucketStatus.PRESERVED
    assert pipeline.bucket_can_be_released[(0, False)] is False


def test_cuda_graph_pinned_lazy_release_keeps_communicating_status():
    pipeline, weight_buffer = _make_pipeline(status=BucketStatus.COMMUNICATING)

    pipeline.pin_cuda_graph_bucket(0, bwd=False)
    pipeline.release_bucket(0, bwd=False, lazy=True)

    assert not weight_buffer.freed
    assert pipeline.bucket_status[(0, False)] == BucketStatus.COMMUNICATING
    assert pipeline.bucket_can_be_released[(0, False)] is False


def test_cuda_graph_pinned_reset_preserves_unit_bucket():
    pipeline, weight_buffer = _make_pipeline()

    pipeline.pin_cuda_graph_bucket(0, bwd=False)
    pipeline.reset(preserve_non_fsdp_units=False)

    assert not weight_buffer.freed
    assert pipeline.bucket_status[(0, False)] == BucketStatus.PRESERVED
    assert pipeline.bucket_can_be_released[(0, False)] is False


def test_cuda_graph_pinned_train_eval_train_reset_preserves_storage():
    pipeline, weight_buffer = _make_pipeline()

    pipeline.pin_cuda_graph_bucket(0, bwd=False)

    # Training release hook runs after the graph-covered module. It must not
    # return the captured bucket storage to the double-buffer allocator.
    pipeline.release_bucket(0, bwd=False, lazy=True)

    # Evaluation uses the same FSDP hooks and synchronizes the all-gather
    # pipeline afterward. This is the path most likely to perturb allocator
    # order when graph buckets are not pinned.
    pipeline.bucket_can_be_released[(0, False)] = True
    pipeline.reset(preserve_non_fsdp_units=True)

    assert not weight_buffer.freed
    assert pipeline.bucket_status[(0, False)] == BucketStatus.PRESERVED
    assert pipeline.bucket_can_be_released[(0, False)] is False


def test_cuda_graph_pinned_bucket_keeps_allocator_slot_across_train_eval_train():
    pipeline, weight_buffer = _make_pipeline()

    captured_address = weight_buffer.address
    pipeline.pin_cuda_graph_bucket(0, bwd=False)

    # Training release marks the bucket reusable, and an eval/reset cycle would
    # normally recycle it before training resumes.
    pipeline.release_bucket(0, bwd=False, lazy=True)
    pipeline.bucket_can_be_released[(0, False)] = True
    pipeline.reset(preserve_non_fsdp_units=True)

    # Simulate another allocation that happens during eval. If the graph bucket
    # had been freed, the fake allocator would hand out captured_address here.
    eval_address = weight_buffer.allocator.allocate("eval_bucket")

    assert weight_buffer.address == captured_address
    assert eval_address != captured_address
    assert pipeline.bucket_status[(0, False)] == BucketStatus.PRESERVED
    assert pipeline.bucket_can_be_released[(0, False)] is False


def test_clearing_cuda_graph_pin_reenables_release():
    pipeline, weight_buffer = _make_pipeline()

    pipeline.pin_cuda_graph_bucket(0, bwd=False)
    pipeline.clear_cuda_graph_pinned_buckets()
    pipeline.release_bucket(0, bwd=False)

    assert weight_buffer.freed
    assert pipeline.bucket_status[(0, False)] == BucketStatus.EMPTY
