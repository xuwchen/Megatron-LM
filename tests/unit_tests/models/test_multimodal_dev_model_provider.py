# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the standalone multimodal model provider."""

from types import SimpleNamespace

from examples.multimodal_dev import pretrain_multimodal
from examples.multimodal_dev.models import MODEL_REGISTRY


def test_model_provider_propagates_runtime_tp_to_vision(monkeypatch):
    """Vision TP metadata must match the process group used by its modules."""
    args = SimpleNamespace(
        model_arch="test_arch", vision_num_layers=2, model_variant="test", recompute_vision=False
    )
    language_config = SimpleNamespace(bf16=True, fp16=False, tensor_model_parallel_size=2)
    vision_config = SimpleNamespace(
        bf16=False, fp16=False, tensor_model_parallel_size=1, context_parallel_size=1
    )
    built_model = object()
    captured = {}

    def vision_config_fn(**kwargs):
        captured["vision_config_kwargs"] = kwargs
        return vision_config

    def model_factory_fn(**kwargs):
        captured["model_factory_kwargs"] = kwargs
        return built_model

    monkeypatch.setattr(pretrain_multimodal, "get_args", lambda: args)
    monkeypatch.setattr(
        pretrain_multimodal, "core_transformer_config_from_args", lambda _args: language_config
    )
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "test_arch",
        {"vision_config_fn": vision_config_fn, "model_factory_fn": model_factory_fn},
    )

    result = pretrain_multimodal.model_provider()

    assert result is built_model
    assert vision_config.tensor_model_parallel_size == 2
    assert vision_config.context_parallel_size == 1
    assert captured["vision_config_kwargs"] == {"num_layers_override": 2, "variant": "test"}
    assert captured["model_factory_kwargs"]["vision_config"] is vision_config
