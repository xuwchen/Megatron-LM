# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    _GTP_TE_REQUIRED_HOOKS,
    _validate_te_gtp_protocol,
    _validate_te_gtp_version,
)


@pytest.mark.parametrize(
    ("version", "compatible"),
    [
        ("2.16.9", False),
        ("2.17.0.dev0", True),
        ("2.17.0rc1", True),
        ("2.17", True),
        ("2.18.0.dev0", True),
    ],
)
def test_gtp_te_version_gate(version, compatible):
    """Exercise the production gate, not a re-implementation of the comparison."""
    if compatible:
        _validate_te_gtp_version(version)
    else:
        with pytest.raises(ImportError, match="requires TransformerEngine"):
            _validate_te_gtp_version(version)


@pytest.mark.parametrize("version", ["2.17.0-nvidia-gtp", "not-a-version", None])
def test_gtp_te_version_gate_degrades_on_unparseable_versions(version):
    """Odd version tags must raise ImportError (caught upstream -> HAVE_GTP=False), not crash."""
    with pytest.raises(ImportError, match="Cannot parse"):
        _validate_te_gtp_version(version)


def test_gtp_te_protocol_accepts_complete_hook_registry():
    module = SimpleNamespace(**{hook: object() for hook in _GTP_TE_REQUIRED_HOOKS})
    _validate_te_gtp_protocol(module)


@pytest.mark.parametrize("missing_hook", _GTP_TE_REQUIRED_HOOKS)
def test_gtp_te_protocol_rejects_missing_hook(missing_hook):
    """A version string alone must not admit a TE build that lacks a required hook."""
    hooks = {hook: object() for hook in _GTP_TE_REQUIRED_HOOKS if hook != missing_hook}
    with pytest.raises(ImportError, match=missing_hook):
        _validate_te_gtp_protocol(SimpleNamespace(**hooks))


def test_required_hooks_exist_in_installed_te():
    """Anchor the hook list to reality: every listed hook must exist in a GTP-capable TE.

    Skips on stock TE builds without the companion distributed-weight module; on companion
    builds this catches a hook rename that the SimpleNamespace tests above cannot see.
    """
    dw = pytest.importorskip("transformer_engine.pytorch.distributed_weight")
    missing = [hook for hook in _GTP_TE_REQUIRED_HOOKS if not hasattr(dw, hook)]
    assert not missing, f"installed TE lacks GTP hooks: {missing}"
