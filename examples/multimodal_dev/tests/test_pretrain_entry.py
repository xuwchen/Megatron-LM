# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the pretrain_multimodal entry configuration."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.pretrain_multimodal import (
    configure_vision_recompute,
    validate_entry_args,
)


def test_recompute_vision_uses_one_whole_tower_block():
    # Whole-tower contract: full recompute must configure ONE uniform block
    # spanning every layer, so only the patch-embed output is saved. A
    # one-layer block size (the value this entry started from) saves every
    # layer's input and dominates vision memory at heavy payloads — invisible
    # to CPU tests and to the 4K smoke, it only shows up as a 128K OOM.
    vision_config = SimpleNamespace(
        num_layers=24, recompute_granularity=None, recompute_method=None, recompute_num_layers=None
    )
    configure_vision_recompute(vision_config)
    assert (
        vision_config.recompute_granularity,
        vision_config.recompute_method,
        vision_config.recompute_num_layers,
    ) == ("full", "uniform", 24)


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
