# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest

import megatron.core.transformer.mla_qk_norm_config as qk_norm_config
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mla_qk_norm_config import QKNormConfigResolver
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig


class _Linear:
    pass


class _FusedNormLinear:
    pass


class _Norm:
    pass


class _Backend:
    @staticmethod
    def layer_norm(**_kwargs):
        return _Norm

    @staticmethod
    def column_parallel_linear():
        return _Linear

    @staticmethod
    def column_parallel_layer_norm_linear():
        return _FusedNormLinear


def _make_config(*, q_lora_rank=32):
    return MLATransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=1,
        q_lora_rank=q_lora_rank,
        qk_layernorm=True,
        layernorm_epsilon=1.0e-5,
    )


def _make_submodules(**overrides):
    values = dict(
        linear_q_proj=None,
        linear_q_up_proj=None,
        linear_kv_up_proj=None,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _mock_backend(monkeypatch):
    monkeypatch.setattr(qk_norm_config, "get_backend", lambda _impl: _Backend())


def test_fused_norm_upgrade_keeps_linear_spec_params():
    """Upgrading a plain-linear spec to the fused norm+linear keeps its params (GTP opt-out)."""
    config = _make_config()
    linear_spec = ModuleSpec(module=_Linear, params={"gtp_remat_opt_out": True})
    submodules = _make_submodules(linear_q_up_proj=linear_spec, linear_kv_up_proj=linear_spec)

    resolved = QKNormConfigResolver(config, submodules).resolve()

    for projection in ("linear_q_up_proj", "linear_kv_up_proj"):
        spec = resolved[projection]
        assert spec.module is _FusedNormLinear
        assert spec.params == {"gtp_remat_opt_out": True}
    assert linear_spec.params == {"gtp_remat_opt_out": True}
