# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the pretrain_multimodal entry configuration."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.pretrain_multimodal import (
    validate_entry_args,
)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        # The model factory happily builds an mtp_block_spec, so this entry
        # rejection is the only thing standing between an MTP run and a
        # silently unwired model.
        ({"mtp_num_layers": 1}, "MTP is not wired through"),
        # forward_step keeps a runtime guard as defense in depth; this one
        # fails in seconds instead of after multi-node model construction.
        ({"cuda_graph_impl": "local"}, "cuda-graph-impl"),
    ],
)
def test_entry_rejects_unsupported_configurations(overrides, message):
    base = dict(
        pipeline_model_parallel_size=1,
        mtp_num_layers=0,
        use_packed_sequence=True,
        cuda_graph_impl="none",
    )
    args = SimpleNamespace(**{**base, **overrides})
    with pytest.raises(ValueError, match=message):
        validate_entry_args(args)
