# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from examples.multimodal_dev.models.qwen35_vl import factory


def test_qwen_language_spec_composes_engram(monkeypatch):
    raw_spec = object()
    engram_config = object()
    wrapped_spec = object()
    args = SimpleNamespace()
    language_config = object()

    monkeypatch.setattr(
        factory,
        "get_qwen35_vl_language_spec",
        lambda **kwargs: raw_spec,
    )
    monkeypatch.setattr(
        factory,
        "EngramConfig",
        SimpleNamespace(from_args=lambda actual_args, actual_config: engram_config),
    )

    def apply_engram(actual_spec, actual_engram_config):
        assert actual_spec is raw_spec
        assert actual_engram_config is engram_config
        return wrapped_spec

    monkeypatch.setattr(factory, "apply_engram_to_layer_spec", apply_engram)

    result = factory._build_language_spec(args, language_config, vp_stage=None)

    assert result is wrapped_spec


def test_qwen_language_spec_is_unchanged_without_engram(monkeypatch):
    raw_spec = object()
    args = SimpleNamespace()
    language_config = object()

    monkeypatch.setattr(
        factory,
        "get_qwen35_vl_language_spec",
        lambda **kwargs: raw_spec,
    )
    monkeypatch.setattr(
        factory,
        "EngramConfig",
        SimpleNamespace(from_args=lambda actual_args, actual_config: None),
    )

    assert factory._build_language_spec(args, language_config, vp_stage=None) is raw_spec
