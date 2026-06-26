# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Debug tracing and address checks for FSDP buffers used by CUDA graphs.

CUDA graph capture records absolute device addresses.  Megatron-FSDP temporary
communication buffers may be backed by a double-buffer pool whose physical slot
depends on runtime allocate/free order.  This module keeps a lightweight process
local registry so debug builds can turn address drift into a clear error.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GraphBufferKey:
    """Stable logical identity for an FSDP-backed buffer."""

    namespace: str
    kind: str
    bucket_id: int
    is_transpose: bool = False
    allocator_name: str = ""

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class GraphBufferSnapshot:
    """Observed address metadata for a graph buffer at one point in time."""

    key: GraphBufferKey
    event: str
    phase: str
    order: int
    rank: int
    allocated: bool
    data_ptr: Optional[int] = None
    numel: Optional[int] = None
    nbytes: Optional[int] = None
    dtype: Optional[str] = None
    shape: Optional[tuple[int, ...]] = None
    device: Optional[str] = None
    allocator_slot: Optional[Any] = None
    source: Optional[str] = None

    @classmethod
    def from_tensor(
        cls,
        *,
        key: GraphBufferKey,
        event: str,
        phase: str,
        order: int,
        rank: int,
        tensor: torch.Tensor,
        allocator_slot: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> "GraphBufferSnapshot":
        """Build a snapshot from a tensor view."""
        return cls(
            key=key,
            event=event,
            phase=phase,
            order=order,
            rank=rank,
            allocated=True,
            data_ptr=tensor.data_ptr(),
            numel=tensor.numel(),
            nbytes=tensor.numel() * tensor.element_size(),
            dtype=str(tensor.dtype),
            shape=tuple(tensor.shape),
            device=str(tensor.device),
            allocator_slot=_jsonable_slot(allocator_slot),
            source=source,
        )

    @classmethod
    def free_event(
        cls,
        *,
        key: GraphBufferKey,
        phase: str,
        order: int,
        rank: int,
        allocator_slot: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> "GraphBufferSnapshot":
        """Build a logical free snapshot."""
        return cls(
            key=key,
            event="free",
            phase=phase,
            order=order,
            rank=rank,
            allocated=False,
            allocator_slot=_jsonable_slot(allocator_slot),
            source=source,
        )

    def to_json(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = dataclasses.asdict(self)
        payload["key"] = self.key.to_json()
        return payload


def _jsonable_slot(slot: Optional[Any]) -> Optional[Any]:
    if slot is None:
        return None
    if isinstance(slot, tuple):
        return list(slot)
    if isinstance(slot, (str, int, float, bool)):
        return slot
    return repr(slot)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class CudaGraphBufferRegistry:
    """Process-local debug registry for FSDP CUDA graph buffer addresses."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all runtime state and close an open trace file."""
        if getattr(self, "_trace_file", None) is not None:
            self._trace_file.close()
        self.enabled = False
        self.assert_addresses_enabled = False
        self.trace_path = None
        self.phase = "eager"
        self.active_capture_stage = None
        self.active_replay_stage = None
        self.order = 0
        self.current: Dict[GraphBufferKey, GraphBufferSnapshot] = {}
        self.capture_candidates: Dict[str, set[GraphBufferKey]] = {}
        self.capture_snapshots: Dict[str, Dict[GraphBufferKey, GraphBufferSnapshot]] = {}
        self.captured: Dict[str, Dict[GraphBufferKey, GraphBufferSnapshot]] = {}
        self._trace_file = None
        self._trace_file_path = None

    def configure(
        self,
        *,
        assert_addresses: bool = False,
        trace_path: Optional[str] = None,
    ) -> None:
        """Enable or update debug behavior."""
        self.assert_addresses_enabled = self.assert_addresses_enabled or assert_addresses
        if trace_path:
            self.trace_path = trace_path
        self.enabled = self.assert_addresses_enabled or self.trace_path is not None

    def begin_capture(self, stage: str) -> None:
        """Start marking allocations as CUDA graph capture candidates."""
        if not self.enabled:
            return
        self.active_capture_stage = stage
        self.capture_candidates.setdefault(stage, set())
        self.capture_snapshots.setdefault(stage, {})
        self.phase = f"capture:{stage}"
        self._write_event({"event": "begin_capture", "stage": stage, "phase": self.phase})

    def finish_capture(self, stage: str) -> None:
        """Freeze captured addresses for allocations observed during capture."""
        if not self.enabled:
            return
        candidates = self.capture_candidates.get(stage, set())
        frozen = self.captured.setdefault(stage, {})
        missing = []
        capture_snapshots = self.capture_snapshots.get(stage, {})
        for key in sorted(candidates, key=repr):
            snapshot = capture_snapshots.get(key)
            if snapshot is None:
                missing.append(key)
                continue
            frozen[key] = snapshot
        self._write_event(
            {
                "event": "finish_capture",
                "stage": stage,
                "captured": len(frozen),
                "missing": [key.to_json() for key in missing],
            }
        )
        if missing:
            logger.warning(
                "Skipped %d CUDA graph FSDP buffer address candidates that were no longer "
                "allocated at capture finish for stage %s.",
                len(missing),
                stage,
            )
        self.active_capture_stage = None
        self.phase = "eager"

    def abort_capture(self, stage: str) -> None:
        """Leave capture mode without freezing addresses."""
        if not self.enabled:
            return
        self._write_event({"event": "abort_capture", "stage": stage})
        self.active_capture_stage = None
        self.phase = "eager"

    def begin_replay(self, stage: str) -> None:
        """Mark subsequent Python-side allocations as replay preparation."""
        if not self.enabled:
            return
        self.active_replay_stage = stage
        self.phase = f"replay:{stage}"

    def finish_replay(self, stage: str) -> None:
        """Leave replay mode."""
        if not self.enabled:
            return
        if self.active_replay_stage == stage:
            self.active_replay_stage = None
        self.phase = "eager"

    def record_allocate(
        self,
        key: GraphBufferKey,
        tensor: torch.Tensor,
        *,
        allocator_slot: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> None:
        """Record that a logical graph buffer currently maps to ``tensor``."""
        if not self.enabled:
            return
        self.order += 1
        snapshot = GraphBufferSnapshot.from_tensor(
            key=key,
            event="allocate",
            phase=self.phase,
            order=self.order,
            rank=_rank(),
            tensor=tensor,
            allocator_slot=allocator_slot,
            source=source,
        )
        self.current[key] = snapshot
        if self.active_capture_stage is not None:
            self.capture_candidates.setdefault(self.active_capture_stage, set()).add(key)
            self.capture_snapshots.setdefault(self.active_capture_stage, {})[key] = snapshot
        self._write_event(snapshot.to_json())

    def record_free(
        self,
        key: GraphBufferKey,
        *,
        allocator_slot: Optional[Any] = None,
        source: Optional[str] = None,
    ) -> None:
        """Record that a logical graph buffer is no longer allocated."""
        if not self.enabled:
            return
        self.order += 1
        snapshot = GraphBufferSnapshot.free_event(
            key=key,
            phase=self.phase,
            order=self.order,
            rank=_rank(),
            allocator_slot=allocator_slot,
            source=source,
        )
        self.current[key] = snapshot
        self._write_event(snapshot.to_json())

    def assert_addresses(self, stage: Optional[str] = None) -> None:
        """Assert current addresses match capture-time addresses."""
        if not self.enabled or not self.assert_addresses_enabled:
            return
        stages = [stage] if stage is not None else list(self.captured)
        errors = []
        for stage_name in stages:
            for key, expected in self.captured.get(stage_name, {}).items():
                actual = self.current.get(key)
                mismatch = _snapshot_mismatch(expected, actual)
                if mismatch is None:
                    continue
                errors.append(_format_mismatch(stage_name, key, expected, actual, mismatch))
        if errors:
            raise RuntimeError(
                "CUDA graph buffer address mismatch:\n"
                + "\n".join(errors)
                + "\nlikely cause: FSDP double buffer allocation order changed between "
                "CUDA graph capture and replay."
            )

    def _trace_output_path(self) -> Optional[Path]:
        if self.trace_path is None:
            return None
        rank = _rank()
        raw_path = self.trace_path.replace("%r", str(rank))
        path = Path(raw_path)
        if "%r" not in self.trace_path and _distributed_world_size() > 1:
            path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
        return path

    def _write_event(self, payload: Dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        if self._trace_file is None:
            path = self._trace_output_path()
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file_path = path
            self._trace_file = open(path, "a", encoding="utf-8")
        payload = dict(payload)
        payload.setdefault("pid", os.getpid())
        payload.setdefault("rank", _rank())
        self._trace_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._trace_file.flush()


def _distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _snapshot_mismatch(
    expected: GraphBufferSnapshot, actual: Optional[GraphBufferSnapshot]
) -> Optional[str]:
    if actual is None:
        return "buffer has not been observed since capture"
    if not actual.allocated:
        return "buffer is currently freed"
    for field_name in ("data_ptr", "nbytes", "dtype", "shape", "device"):
        if getattr(expected, field_name) != getattr(actual, field_name):
            return field_name
    return None


def _format_mismatch(
    stage: str,
    key: GraphBufferKey,
    expected: GraphBufferSnapshot,
    actual: Optional[GraphBufferSnapshot],
    mismatch: str,
) -> str:
    actual_desc = (
        "<missing>"
        if actual is None
        else (
            f"ptr={actual.data_ptr}, dtype={actual.dtype}, shape={actual.shape}, "
            f"nbytes={actual.nbytes}, allocated={actual.allocated}, "
            f"phase={actual.phase}, order={actual.order}, slot={actual.allocator_slot}"
        )
    )
    return (
        f"  stage: {stage}\n"
        f"  bucket_key: {key.to_json()}\n"
        f"  mismatch: {mismatch}\n"
        f"  expected: ptr={expected.data_ptr}, dtype={expected.dtype}, "
        f"shape={expected.shape}, nbytes={expected.nbytes}, "
        f"phase={expected.phase}, order={expected.order}, slot={expected.allocator_slot}\n"
        f"  actual: {actual_desc}"
    )


_REGISTRY = CudaGraphBufferRegistry()
atexit.register(lambda: _REGISTRY.reset())


def reset_cuda_graph_buffer_debug_state() -> None:
    """Reset global debug state. Intended for tests."""
    _REGISTRY.reset()


def configure_cuda_graph_buffer_debug(
    *, assert_addresses: bool = False, trace_path: Optional[str] = None
) -> None:
    """Configure global debug behavior."""
    _REGISTRY.configure(assert_addresses=assert_addresses, trace_path=trace_path)


def configure_cuda_graph_buffer_debug_from_config(config: Any) -> None:
    """Configure global debug behavior from a DDP/FSDP config object."""
    configure_cuda_graph_buffer_debug(
        assert_addresses=getattr(config, "cuda_graph_assert_buffer_addresses", False),
        trace_path=getattr(config, "cuda_graph_buffer_trace_path", None),
    )


def begin_cuda_graph_buffer_capture(stage: str) -> None:
    _REGISTRY.begin_capture(stage)


def finish_cuda_graph_buffer_capture(stage: str) -> None:
    _REGISTRY.finish_capture(stage)


def abort_cuda_graph_buffer_capture(stage: str) -> None:
    _REGISTRY.abort_capture(stage)


def begin_cuda_graph_buffer_replay(stage: str) -> None:
    _REGISTRY.begin_replay(stage)


def finish_cuda_graph_buffer_replay(stage: str) -> None:
    _REGISTRY.finish_replay(stage)


def assert_cuda_graph_buffer_addresses(stage: Optional[str] = None) -> None:
    _REGISTRY.assert_addresses(stage)


def record_cuda_graph_buffer_allocate(
    key: GraphBufferKey,
    tensor: torch.Tensor,
    *,
    allocator_slot: Optional[Any] = None,
    source: Optional[str] = None,
) -> None:
    _REGISTRY.record_allocate(key, tensor, allocator_slot=allocator_slot, source=source)


def record_cuda_graph_buffer_free(
    key: GraphBufferKey,
    *,
    allocator_slot: Optional[Any] = None,
    source: Optional[str] = None,
) -> None:
    _REGISTRY.record_free(key, allocator_slot=allocator_slot, source=source)

