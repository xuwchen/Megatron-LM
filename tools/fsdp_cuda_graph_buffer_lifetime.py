# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Analyze Megatron-FSDP CUDA graph buffer lifetime JSONL traces.

The trace is emitted by ``--cuda-graph-buffer-trace-path``.  This tool turns
allocate/free events into logical lifetimes, computes overlap peak live sets,
and derives a deterministic bucket-to-slot plan for fixed-pool allocators.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


@dataclasses.dataclass(frozen=True)
class LogicalKey:
    """Logical identity of one traced graph buffer."""

    rank: int
    namespace: str
    kind: str
    bucket_id: int
    is_transpose: bool
    allocator_name: str

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "LogicalKey":
        return cls.from_payload(event["key"], rank=int(event.get("rank", 0)))

    @classmethod
    def from_payload(cls, key: dict[str, Any], *, rank: int) -> "LogicalKey":
        return cls(
            rank=rank,
            namespace=str(key.get("namespace", "")),
            kind=str(key.get("kind", "")),
            bucket_id=int(key.get("bucket_id", -1)),
            is_transpose=bool(key.get("is_transpose", False)),
            allocator_name=str(key.get("allocator_name", "")),
        )

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Interval:
    """One logical allocate/free lifetime."""

    key: LogicalKey
    start: int
    end: int
    nbytes: int
    dtype: str
    shape: Any
    device: str
    bucket_offset: int | None
    observed_slot: int | None
    start_phase: str
    end_phase: str

    def overlaps(self, other: "Interval") -> bool:
        return self.start < other.end and other.start < self.end


def _parse_slot(slot: Any) -> tuple[int | None, int | None]:
    if isinstance(slot, list) and len(slot) == 2:
        return int(slot[0]), int(slot[1])
    if isinstance(slot, tuple) and len(slot) == 2:
        return int(slot[0]), int(slot[1])
    return None, None


def _iter_events(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL event") from exc
                event["_source_path"] = str(path)
                event["_source_line"] = line_no
                yield event


def read_captured_keys(paths: Iterable[Path], stage: str) -> set[LogicalKey]:
    """Return keys frozen at CUDA graph capture finish for ``stage``."""
    captured: set[LogicalKey] = set()
    for event in _iter_events(paths):
        if event.get("event") != "finish_capture" or event.get("stage") != stage:
            continue
        rank = int(event.get("rank", 0))
        for key in event.get("captured_keys", []):
            captured.add(LogicalKey.from_payload(key, rank=rank))
    return captured


def build_intervals(paths: Iterable[Path]) -> tuple[list[Interval], dict[str, int]]:
    """Build logical intervals from allocate/free events."""
    live: dict[LogicalKey, dict[str, Any]] = {}
    intervals: list[Interval] = []
    stats = defaultdict(int)
    last_order = 0

    for event in sorted(_iter_events(paths), key=lambda item: (int(item.get("rank", 0)), int(item.get("order", 0)))):
        if "key" not in event or event.get("event") not in {"allocate", "free"}:
            continue
        key = LogicalKey.from_event(event)
        order = int(event.get("order", 0))
        last_order = max(last_order, order)
        if event["event"] == "allocate":
            if key in live:
                stats["duplicate_allocate_without_free"] += 1
                continue
            observed_slot, bucket_offset = _parse_slot(event.get("allocator_slot"))
            live[key] = {
                "start": order,
                "nbytes": int(event.get("nbytes") or 0),
                "dtype": str(event.get("dtype") or ""),
                "shape": event.get("shape"),
                "device": str(event.get("device") or ""),
                "bucket_offset": bucket_offset,
                "observed_slot": observed_slot,
                "start_phase": str(event.get("phase") or ""),
            }
            stats["allocates"] += 1
            continue

        if key not in live:
            stats["free_without_allocate"] += 1
            continue
        opened = live.pop(key)
        intervals.append(
            Interval(
                key=key,
                start=opened["start"],
                end=order,
                nbytes=opened["nbytes"],
                dtype=opened["dtype"],
                shape=opened["shape"],
                device=opened["device"],
                bucket_offset=opened["bucket_offset"],
                observed_slot=opened["observed_slot"],
                start_phase=opened["start_phase"],
                end_phase=str(event.get("phase") or ""),
            )
        )
        stats["frees"] += 1

    eof_order = last_order + 1
    for key, opened in sorted(live.items(), key=lambda item: repr(item[0])):
        intervals.append(
            Interval(
                key=key,
                start=opened["start"],
                end=eof_order,
                nbytes=opened["nbytes"],
                dtype=opened["dtype"],
                shape=opened["shape"],
                device=opened["device"],
                bucket_offset=opened["bucket_offset"],
                observed_slot=opened["observed_slot"],
                start_phase=opened["start_phase"],
                end_phase="eof",
            )
        )
        stats["open_at_eof"] += 1

    stats["intervals"] = len(intervals)
    return intervals, dict(stats)


def _group_key(interval: Interval) -> tuple[int, str, str, str, int | None]:
    return (
        interval.key.rank,
        interval.key.namespace,
        interval.key.kind,
        interval.key.allocator_name,
        interval.bucket_offset,
    )


def _peak_live(intervals: list[Interval]) -> tuple[int, int]:
    events: list[tuple[int, int, int]] = []
    for interval in intervals:
        events.append((interval.start, 1, interval.nbytes))
        events.append((interval.end, -1, -interval.nbytes))
    live = 0
    live_bytes = 0
    peak_live = 0
    peak_bytes = 0
    for _, delta_live, delta_bytes in sorted(events, key=lambda item: (item[0], item[1])):
        live += delta_live
        live_bytes += delta_bytes
        peak_live = max(peak_live, live)
        peak_bytes = max(peak_bytes, live_bytes)
    return peak_live, peak_bytes


def _build_conflicts(intervals: list[Interval]) -> dict[LogicalKey, set[LogicalKey]]:
    conflicts: dict[LogicalKey, set[LogicalKey]] = defaultdict(set)
    active: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: (item.start, item.end, repr(item.key))):
        active = [item for item in active if item.end > interval.start]
        conflicts.setdefault(interval.key, set())
        for other in active:
            if interval.key == other.key:
                continue
            if not interval.overlaps(other):
                continue
            conflicts[interval.key].add(other.key)
            conflicts[other.key].add(interval.key)
        active.append(interval)
    return conflicts


def _color_conflict_graph(conflicts: dict[LogicalKey, set[LogicalKey]]) -> dict[LogicalKey, int]:
    colors: dict[LogicalKey, int] = {}
    nodes = sorted(conflicts, key=lambda key: (-len(conflicts[key]), repr(key)))
    for node in nodes:
        used = {colors[neighbor] for neighbor in conflicts[node] if neighbor in colors}
        color = 0
        while color in used:
            color += 1
        colors[node] = color
    return colors


def derive_plan(intervals: list[Interval], pool_size: int) -> dict[str, Any]:
    """Derive a fixed-pool slot plan from observed lifetimes."""
    by_group: dict[tuple[int, str, str, str, int | None], list[Interval]] = defaultdict(list)
    for interval in intervals:
        if interval.bucket_offset is None:
            continue
        by_group[_group_key(interval)].append(interval)

    groups = []
    total_overflow_groups = 0
    for group_key, group_intervals in sorted(by_group.items(), key=lambda item: repr(item[0])):
        rank, namespace, kind, allocator_name, bucket_offset = group_key
        peak_live, peak_bytes = _peak_live(group_intervals)
        conflicts = _build_conflicts(group_intervals)
        colors = _color_conflict_graph(conflicts)
        colors_used = max(colors.values(), default=-1) + 1
        overflow = colors_used > pool_size
        total_overflow_groups += int(overflow)

        intervals_by_key: dict[LogicalKey, list[Interval]] = defaultdict(list)
        for interval in group_intervals:
            intervals_by_key[interval.key].append(interval)

        assignments = []
        for key, color in sorted(colors.items(), key=lambda item: (item[1], repr(item[0]))):
            key_intervals = intervals_by_key[key]
            max_nbytes = max(interval.nbytes for interval in key_intervals)
            assignments.append(
                {
                    "key": key.to_json(),
                    "planned_slot": [color, bucket_offset],
                    "color": color,
                    "bucket_offset": bucket_offset,
                    "interval_count": len(key_intervals),
                    "first_start": min(interval.start for interval in key_intervals),
                    "last_end": max(interval.end for interval in key_intervals),
                    "max_nbytes": max_nbytes,
                    "dtype": key_intervals[0].dtype,
                    "shape": key_intervals[0].shape,
                    "device": key_intervals[0].device,
                    "observed_slots": sorted(
                        {
                            interval.observed_slot
                            for interval in key_intervals
                            if interval.observed_slot is not None
                        }
                    ),
                }
            )

        groups.append(
            {
                "rank": rank,
                "namespace": namespace,
                "kind": kind,
                "allocator_name": allocator_name,
                "bucket_offset": bucket_offset,
                "peak_live": peak_live,
                "peak_live_bytes": peak_bytes,
                "colors_used": colors_used,
                "pool_size": pool_size,
                "overflow": overflow,
                "assignments": assignments,
            }
        )

    return {
        "version": 1,
        "pool_size": pool_size,
        "summary": {
            "groups": len(groups),
            "overflow_groups": total_overflow_groups,
            "intervals_with_slots": sum(len(item) for item in by_group.values()),
        },
        "groups": groups,
    }


def _print_summary(plan: dict[str, Any], stats: dict[str, int]) -> None:
    print("Trace stats:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print("Plan groups:")
    for group in plan["groups"]:
        overflow = " OVERFLOW" if group["overflow"] else ""
        print(
            "  "
            f"rank={group['rank']} kind={group['kind']} allocator={group['allocator_name']} "
            f"offset={group['bucket_offset']} peak_live={group['peak_live']} "
            f"colors={group['colors_used']}/{group['pool_size']}"
            f"{overflow}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path, help="Trace JSONL files")
    parser.add_argument("--pool-size", type=int, default=2, help="Fixed-pool buffer groups")
    parser.add_argument(
        "--captured-stage",
        type=str,
        default=None,
        help="Only derive a plan for keys frozen by the named CUDA graph capture stage.",
    )
    parser.add_argument("--output-plan", type=Path, default=None, help="Write derived plan JSON")
    args = parser.parse_args()

    intervals, stats = build_intervals(args.jsonl)
    if args.captured_stage is not None:
        captured = read_captured_keys(args.jsonl, args.captured_stage)
        stats["captured_stage_keys"] = len(captured)
        intervals = [interval for interval in intervals if interval.key in captured]
        stats["intervals_after_captured_stage_filter"] = len(intervals)
    plan = derive_plan(intervals, args.pool_size)
    plan["source_files"] = [str(path) for path in args.jsonl]
    plan["captured_stage"] = args.captured_stage
    _print_summary(plan, stats)

    if args.output_plan is not None:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        with args.output_plan.open("w", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
