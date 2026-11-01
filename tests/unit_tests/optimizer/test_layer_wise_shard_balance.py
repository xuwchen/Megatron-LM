# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cost-balanced whole-param ownership of LayerWiseDistributedOptimizer (no layout path).

CPU-only: the optimizer object is a bare skeleton and the process-group helpers are faked, so
only the assignment logic (_shard_params_cost_balanced / _step_cost) is exercised.
"""

import torch

import megatron.core.optimizer.layer_wise_optimizer as lwo_mod
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer as LWO

GTP = 8


def _matrix(rows, cols, **attrs):
    p = torch.nn.Parameter(torch.empty(rows, cols, dtype=torch.bfloat16))
    p.is_gtp_weight_remat = True
    p.gtp_remat_size = GTP
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


def _scalar_param(*shape, **attrs):
    p = torch.nn.Parameter(torch.empty(*shape, dtype=torch.bfloat16))
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


class _Opt:
    def __init__(self, params, **group_flags):
        self.param_groups = [dict(params=list(params), **group_flags)]


def _shard(monkeypatch, optimizers, dp_size, rank, expt_dp_size=1):
    lwo = object.__new__(LWO)
    lwo.dp_cp = object()
    lwo.expt_dp = object()
    sizes = {id(lwo.dp_cp): dp_size, id(lwo.expt_dp): expt_dp_size}
    monkeypatch.setattr(lwo_mod, "get_pg_size", lambda g: sizes[id(g)])
    monkeypatch.setattr(lwo_mod, "get_pg_rank", lambda g: rank)
    lwo._shard_params_cost_balanced(optimizers, dp_size, expt_dp_size, model_chunks=None)
    return lwo


def _kimi_like_dense():
    """39 equal attention matrices, one dense-MLP fc1 (11x their cost) and fc2 (5.3x)."""
    big = [_matrix(3, 14) for _ in range(39)]  # full [24, 14]: cost 24*14*14
    fc1 = _matrix(33, 14)  # full [264, 14]
    fc2 = _matrix(16, 14)  # full [128, 14]
    return big, fc1, fc2


def test_step_cost_uses_full_gtp_shape_for_muon_and_numel_for_scalars():
    assert LWO._step_cost(_matrix(3, 14)) == 24 * 14 * 14
    plain = _scalar_param(3, 14)  # 2-D but not GTP: still a Muon matrix
    assert LWO._step_cost(plain) == 14 * 3 * 3
    vec = _scalar_param(100)
    assert LWO._step_cost(vec) == 100 * LWO._SCALAR_COST_PER_ELEMENT
    emb = _scalar_param(10, 20, is_embedding_or_output_parameter=True)
    assert LWO._step_cost(emb) == 200 * LWO._SCALAR_COST_PER_ELEMENT
    router = _scalar_param(10, 20, use_muon=False)
    assert LWO._step_cost(router) == 200 * LWO._SCALAR_COST_PER_ELEMENT


def test_dense_mlp_does_not_pile_onto_the_rank_that_already_holds_its_share(monkeypatch):
    """Ping-pong by numel gave every rank ~10 attention matrices and then rank 0 the fc1 and
    rank 1 the fc2 on top (cost +110% / +53%). Cost balancing keeps every rank within a
    matrix of the mean."""
    big, fc1, fc2 = _kimi_like_dense()
    params = big + [fc1, fc2]
    lwo = _shard(monkeypatch, [_Opt(params)], dp_size=4, rank=0)
    loads = [sum(LWO._step_cost(p) for p in owned) for owned in lwo.dp_cp_params_list]
    mean = sum(loads) / 4
    unit = LWO._step_cost(big[0])
    assert max(loads) - mean <= unit, loads
    assert mean - min(loads) <= unit, loads
    owner_of = {id(p): r for r, owned in enumerate(lwo.dp_cp_params_list) for p in owned}
    assert owner_of[id(fc1)] != owner_of[id(fc2)]
    # Each rank owns every param exactly once, and the local group holds rank 0's share.
    assert sorted(owner_of) == sorted(id(p) for p in params)
    assert sorted(id(p) for p in lwo.dp_cp_params_list[0]) == sorted(
        id(p) for p in params if owner_of[id(p)] == 0
    )


def test_assignment_is_identical_on_every_rank_and_local_groups_match(monkeypatch):
    big, fc1, fc2 = _kimi_like_dense()
    emb = _scalar_param(50, 14, is_embedding_or_output_parameter=True)
    norms = [_scalar_param(14) for _ in range(5)]
    ownerships = []
    for rank in range(4):
        muon = _Opt(big + [fc1, fc2])
        adam = _Opt([emb] + norms)
        lwo = _shard(monkeypatch, [muon, adam], dp_size=4, rank=rank)
        ownerships.append([[id(p) for p in owned] for owned in lwo.dp_cp_params_list])
        # The narrowed groups contain exactly this rank's params, in group order.
        assert [id(p) for p in muon.param_groups[0]["params"]] == [
            id(p) for p in big + [fc1, fc2] if id(p) in ownerships[-1][rank]
        ]
        assert [id(p) for p in adam.param_groups[0]["params"]] == [
            id(p) for p in [emb] + norms if id(p) in ownerships[-1][rank]
        ]
    assert all(o == ownerships[0] for o in ownerships)


def test_scalar_params_are_charged_so_the_embedding_avoids_the_heaviest_rank(monkeypatch):
    """An offloaded embedding's Adam step costs real time; it must not be stacked on top of
    the rank that already carries the most Newton-Schulz work."""
    big, fc1, fc2 = _kimi_like_dense()
    emb = _scalar_param(2, 7, is_embedding_or_output_parameter=True)  # 14 el x 4096 ~ 12 matrices
    assert LWO._step_cost(emb) > LWO._step_cost(fc1)  # the embedding is the costliest item
    lwo = _shard(monkeypatch, [_Opt(big + [fc1, fc2]), _Opt([emb])], dp_size=4, rank=0)
    owner_of = {id(p): r for r, owned in enumerate(lwo.dp_cp_params_list) for p in owned}
    assert owner_of[id(emb)] != owner_of[id(fc1)]
    loads = [sum(LWO._step_cost(p) for p in owned) for owned in lwo.dp_cp_params_list]
    assert max(loads) - min(loads) <= LWO._step_cost(fc1), loads


def test_expert_groups_balance_on_their_own_axis(monkeypatch):
    experts = [_matrix(4, 6) for _ in range(6)]
    dense = [_matrix(3, 14) for _ in range(4)]
    lwo = _shard(
        monkeypatch,
        [_Opt(dense), _Opt(experts, is_expert_parallel=True)],
        dp_size=4,
        rank=0,
        expt_dp_size=2,
    )
    assert [len(owned) for owned in lwo.dp_cp_params_list] == [1, 1, 1, 1]
    assert [len(owned) for owned in lwo.expt_dp_params_list] == [3, 3]
    expert_ids = {id(p) for p in experts}
    assert all(id(p) in expert_ids for owned in lwo.expt_dp_params_list for p in owned)
