# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for GTP weight and token layouts."""

import itertools
from fractions import Fraction
from types import SimpleNamespace

import pytest

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.gtp_parallel_layout import (
    GTPParallelLayout,
    get_batch_parallel_size,
    resolve_tensor_parallel_sequence_shards,
)


def _by_rank(groups):
    return {rank: group for group in groups for rank in group}


class TestGTPParallelLayout:
    def test_target_nvlink_placement(self):
        for tp in (1, 2):
            _, num_sequence_shards = resolve_tensor_parallel_sequence_shards(
                tp, 64, 64 // tp, tp > 1
            )
            layout = GTPParallelLayout(128, tp, 1, 2, 64 // tp, num_sequence_shards)
            assert layout.batch_parallel_size == 1
            assert layout.dp == 1
            assert layout.num_weight_replicas == 2
            for group in layout.weights.get_ranks("tp-gtp_remat"):
                assert len(group) == 64
                assert len({rank // 64 for rank in group}) == 1
            assert all(len(g) == 128 // tp for g in layout.data.get_ranks("cp"))

    def test_complete_orthogonal_contribution_domains(self):
        # GTP sequence and sample partitions can both vary independently of CP.
        for tp, pp, cp, num_sequence_shards, extra_data, dp in itertools.product(
            (1, 2), (1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 3), (1, 2)
        ):
            weight = num_sequence_shards * extra_data
            world = tp * pp * cp * weight * dp
            layout = GTPParallelLayout(
                world, tp, pp, cp, weight, num_sequence_shards, rank_offset=7
            )
            weights = _by_rank(layout.weights.get_ranks("gtp_remat"))
            replicas = _by_rank(layout.weights.get_ranks("dp"))
            contexts = _by_rank(layout.data.get_ranks("cp"))
            samples = _by_rank(layout.data.get_ranks("gtp_remat-dp"))
            tokens = _by_rank(layout.data.get_ranks("cp-gtp_remat-dp"))
            for rank in range(7, 7 + world):
                assert set(weights[rank]) & set(replicas[rank]) == {rank}
                assert set(contexts[rank]) & set(samples[rank]) == {rank}
                assert len(contexts[rank]) == cp * num_sequence_shards
                assert len(weights[rank]) == weight
                contributions = [peer for r in replicas[rank] for peer in weights[r]]
                assert len(contributions) == len(set(contributions))
                assert set(contributions) == set(tokens[rank])
                # Replica peers own the same shard; CP peers process the same sequence.
                assert len({weights[r].index(r) for r in replicas[rank]}) == 1
                assert len({samples[r].index(r) for r in contexts[rank]}) == 1

    def test_cp_extension_preserves_weight_and_optimizer_ownership(self):
        for tp in (1, 2):
            layouts = [
                GTPParallelLayout(256, tp, 1, cp, 64 // tp, num_sequence_shards)
                for cp in (1, 2, 4)
                for num_sequence_shards in (2, 8, 32)
            ]
            for layout in layouts[1:]:
                for group in ("gtp_remat", "dp", "tp-gtp_remat-pp"):
                    assert layouts[0].weights.get_ranks(group) == layout.weights.get_ranks(group)

    def test_gradient_reductions_count_each_contribution_once(self):
        for cp, num_sequence_shards, weight in ((2, 4, 4), (2, 2, 4), (3, 1, 4), (1, 1, 4)):
            world = cp * weight * 2
            layout = GTPParallelLayout(world, 1, 1, cp, weight, num_sequence_shards)
            weights = _by_rank(layout.weights.get_ranks("gtp_remat"))
            replicas = _by_rank(layout.weights.get_ranks("dp"))
            # Rank-distinct values detect accidentally reduced unrelated shards.
            gradients = [
                [Fraction((r + 1) * (s + 3), 7) for s in range(weight)] for r in range(world)
            ]
            for per_token in (False, True):
                w_div = 1 if per_token else weight
                r_div = 1 if per_token else layout.num_weight_replicas
                token_div = sum(range(1, world + 1)) if per_token else 1
                for rank in range(world):
                    shard = weights[rank].index(rank)
                    matrix_grad = (
                        sum(
                            sum(gradients[p][shard] for p in weights[r]) / w_div
                            for r in replicas[rank]
                        )
                        / r_div
                        / token_div
                    )
                    # Norm/bias: DDP across replicas followed by GTP all-reduce.
                    replicated_grad = (
                        sum(
                            sum(gradients[p][0] for p in replicas[r]) / r_div for r in weights[rank]
                        )
                        / w_div
                        / token_div
                    )
                    divisor = token_div if per_token else world
                    assert matrix_grad == sum(g[shard] for g in gradients) / divisor
                    assert replicated_grad == sum(g[0] for g in gradients) / divisor

    def test_sample_accounting(self):
        legacy = SimpleNamespace(data_parallel_size=2, gtp_weight_remat_size=4)
        assert get_batch_parallel_size(legacy) == 8
        overlap = SimpleNamespace(
            data_parallel_size=1, gtp_weight_remat_size=32, gtp_remat_num_sequence_shards=32
        )
        assert get_batch_parallel_size(overlap) == 1
        overlap.batch_parallel_size = 3
        assert get_batch_parallel_size(overlap) == 3

    @pytest.mark.parametrize(
        "tp,sp,shards,num_sequence_shards",
        [(1, False, 64, 64), (2, True, 64, 32), (2, True, 16, 8)],
    )
    def test_sequence_shards_include_sp(self, tp, sp, shards, num_sequence_shards):
        config = ModelParallelConfig(
            tensor_model_parallel_size=tp,
            sequence_parallel=sp,
            tensor_parallel_num_weight_shards=64,
            tensor_parallel_num_sequence_shards=shards,
            context_parallel_size=2 * num_sequence_shards,
        )
        assert config.gtp_remat_num_sequence_shards == num_sequence_shards
        layout = GTPParallelLayout(128, tp, 1, 2, 64 // tp, num_sequence_shards)
        assert layout.dp == 1
        assert layout.batch_parallel_size == (64 // tp) // num_sequence_shards

    @pytest.mark.parametrize("tp,sp", [(1, False), (2, False), (2, True)])
    def test_default_has_one_gtp_sequence_shard(self, tp, sp):
        shards, num_sequence_shards = resolve_tensor_parallel_sequence_shards(tp, None, 4, sp)
        assert shards == (tp if sp else 1)
        assert num_sequence_shards == 1
        config = ModelParallelConfig(
            tensor_model_parallel_size=tp,
            sequence_parallel=sp,
            tensor_parallel_num_weight_shards=4 * tp,
            context_parallel_size=2,
        )
        assert config.gtp_remat_num_sequence_shards == 1
        assert config.context_parallel_size == 2


class TestPackingSchedulerSlot:
    """The packing scheduler's DP x CP slot must follow the runtime CP group."""

    @staticmethod
    def _group(ranks, rank):
        return SimpleNamespace(size=lambda: len(ranks), rank=lambda: ranks.index(rank))

    @pytest.mark.parametrize(
        "world,tp,cp,weight,shards",
        [
            (8, 1, 1, 4, 4),  # 2-node proxy: all GTP ranks share one sequence
            (256, 1, 1, 64, 8),  # 128K target: 8 sequence shards x 8 batch shards per group
            (8, 1, 1, 4, 2),  # batch shards without configured CP
            (16, 1, 2, 4, 2),  # configured CP 2 x batch shards 2: DPxCP is not CP-fastest
        ],
    )
    def test_slot_from_dp_and_cp_ranks_follows_runtime_cp_group(
        self, world, tp, cp, weight, shards
    ):
        from megatron.core.datasets.data_schedule_utils import get_scheduler_dp_cp_rank

        layout = GTPParallelLayout(world, tp, 1, cp, weight, num_sequence_shards=shards)
        data = layout.data
        cp_groups = data.get_ranks("cp")
        cp_by_rank = _by_rank(cp_groups)
        dp_by_rank = _by_rank(data.get_ranks("gtp_remat-dp"))
        dp_cp_by_rank = _by_rank(data.get_ranks("gtp_remat-dp-cp"))
        cp_size = cp * shards
        assert all(len(group) == cp_size for group in cp_groups)

        def slot(rank):
            return get_scheduler_dp_cp_rank(
                self._group(dp_by_rank[rank], rank),
                self._group(cp_by_rank[rank], rank),
                self._group(dp_cp_by_rank[rank], rank),
            )

        for dp_cp_ranks in data.get_ranks("gtp_remat-dp-cp"):
            slots = {rank: slot(rank) for rank in dp_cp_ranks}
            # Slots tile the DPxCP grid exactly once.
            assert sorted(slots.values()) == list(range(len(dp_cp_ranks)))
            # Every runtime CP group owns the cp_size consecutive slots of one packed sequence.
            for cp_ranks in cp_groups:
                if cp_ranks[0] not in slots:
                    continue
                sequence_ids = {slots[rank] // cp_size for rank in cp_ranks}
                assert len(sequence_ids) == 1
                assert sorted(slots[rank] % cp_size for rank in cp_ranks) == list(range(cp_size))

        # The positional dp_cp_group.rank() only agrees when the DPxCP group enumerates the
        # runtime CP group contiguously, i.e. when the configured CP is 1 or there is a single
        # batch shard. This is the case the explicit slot exists for.
        positional_matches = all(
            dp_cp_ranks[k : k + cp_size] in cp_groups
            for dp_cp_ranks in data.get_ranks("gtp_remat-dp-cp")
            for k in range(0, len(dp_cp_ranks), cp_size)
        )
        assert positional_matches == (cp == 1 or weight // shards == 1)

    def test_slot_falls_back_to_positional_rank_without_cp_group(self):
        from megatron.core.datasets.data_schedule_utils import get_scheduler_dp_cp_rank

        dp_cp_group = self._group([0, 1, 2, 3], 3)
        assert get_scheduler_dp_cp_rank(self._group([0, 2], 2), None, dp_cp_group) == 3
