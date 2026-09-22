# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Probe: does #7293's load-time shape grant make a non-GTP checkpoint load correctly into a
padded GTP model? Two layouts on world_size=4: TP2 x GTP2 (tp_rank>0 exists) and TP1 x GTP4
(control). Values are the row index, so any offset shift shows up as a value mismatch.

Runs on the gtp-release-dev base WITHOUT the logical-layout fix (818dceb3be); the only
shape-mismatch handling in play is grant_shape_mismatch_for_gtp_padding from #7293, called the
way training/checkpointing.py calls it."""
import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTP_CONFIG,
    update_gtp_config,
    wrap_module_params_gtp,
)
from megatron.core.utils import (
    grant_shape_mismatch_for_gtp_padding,
    make_tp_sharded_tensor_for_checkpoint,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _torchrun_dist_init,
)


@pytest.fixture(scope="module", autouse=True)
def _dist_ready(_torchrun_dist_init):
    yield


def _make_gtp_shard(out_features, in_features, gtp_remat_group):
    class _Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(out_features * in_features, dtype=torch.bfloat16, device="cuda")
                .reshape(out_features, in_features)
            )

    mod = _Dummy()
    wrap_module_params_gtp(mod, ["weight"], gtp_remat_group)
    return mod.weight


def _run(tp, gtp, per_tp, pad_align, tmp_dir, tag, force_grant=False):
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    orig = GTP_CONFIG.pad_for_alignment
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=1, gtp_remat_size=gtp
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'dp_cp', 'dp_cp_gtp_remat']
        )
        gtp_group = ps.get_gtp_weight_remat_group()
        hidden = 4
        tp_rank = ps.get_tensor_model_parallel_rank()
        gtp_rank = ps.get_gtp_weight_remat_rank()
        full = torch.arange(per_tp * tp, dtype=torch.bfloat16, device="cuda").reshape(-1, 1)
        full = full.expand(per_tp * tp, hidden).contiguous()
        my_rows = full[tp_rank * per_tp : (tp_rank + 1) * per_tp].clone()

        with TempNamedDir(tmp_dir / f'probe_{tag}', sync=True) as ckpt_dir:
            update_gtp_config(pad_for_alignment=0)
            plain = {
                "w": make_tp_sharded_tensor_for_checkpoint(
                    tensor=torch.nn.Parameter(my_rows.clone()), key="w", tp_axis=0,
                    prepend_offsets=(), tp_group=pg.tp, dp_cp_group=pg.dp_cp_gtp_remat,
                )
            }
            save(plain, ckpt_dir)

            update_gtp_config(pad_for_alignment=pad_align)
            weight = _make_gtp_shard(per_tp, hidden, gtp_group)
            shard, pad = weight.shape[0], weight.pad_length
            with torch.no_grad():
                weight.zero_()
            target = {
                "w": make_tp_sharded_tensor_for_checkpoint(
                    tensor=weight, key="w", tp_axis=0, prepend_offsets=(),
                    tp_group=pg.tp, dp_cp_group=pg.dp_cp_gtp_remat,
                )
            }
            granted_before = target["w"].allow_shape_mismatch
            # Exactly what training/checkpointing.py does under #7293 before loading.
            grant_shape_mismatch_for_gtp_padding(target, ckpt_dir, pad_align)
            granted_after = target["w"].allow_shape_mismatch
            gpl = getattr(target["w"], "gtp_pad_length", None)
            print(f"[probe {tag}] rank={dist.get_rank()} global={tuple(target['w'].global_shape)} "
                  f"gtp_pad_length_attr={gpl} grant={granted_after}", flush=True)
            if force_grant:
                # Simulate a grant that passes: does the loaded data land on the right rows?
                target["w"].allow_shape_mismatch = True
            loaded = load(target, ckpt_dir)

        got = loaded["w"]
        got = got.data if hasattr(got, "data") else got
        start = gtp_rank * shard
        keep = min(shard, max(0, per_tp - start))
        want = my_rows[start : start + keep]
        ok = torch.equal(got[:keep].cpu(), want.cpu())
        print(
            f"[probe {tag}] rank={dist.get_rank()} tp={tp_rank} gtp={gtp_rank} shard={shard} "
            f"pad={pad} grant {granted_before}->{granted_after} "
            f"want={want[:, 0].tolist()} got={got[:keep, 0].cpu().tolist()} {'OK' if ok else 'SHIFTED'}",
            flush=True,
        )
        assert ok, f"rank={dist.get_rank()} tp={tp_rank} gtp={gtp_rank}: rows shifted"
    finally:
        update_gtp_config(pad_for_alignment=orig)
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _need4():
    if dist.get_world_size() != 4:
        pytest.skip("world_size=4 required")


class TestProbe7293:
    def test_tp2_gtp2_padded_load_from_non_gtp(self, tmp_path_dist_ckpt):
        _need4()
        # per-TP 10 rows, align 3*2=6 -> pad 2, shards of 6; tp_rank 1 exists.
        _run(2, 2, 10, 3, tmp_path_dist_ckpt, "tp2gtp2")

    def test_tp2_gtp2_forced_grant(self, tmp_path_dist_ckpt):
        _need4()
        _run(2, 2, 10, 3, tmp_path_dist_ckpt, "tp2gtp2_forced", force_grant=True)

    def test_tp1_gtp4_padded_load_from_non_gtp_control(self, tmp_path_dist_ckpt):
        _need4()
        # per-TP 10 rows, align 3*4=12 -> pad 2, shards of 3; no tp_rank>0.
        _run(1, 4, 10, 3, tmp_path_dist_ckpt, "tp1gtp4")
