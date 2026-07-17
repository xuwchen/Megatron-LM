# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""train_step forwards p2p_communicator and schedule pg_collection to forward_backward_func."""

import argparse
from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.training import arguments as arguments_mod
from megatron.training import training as training_mod


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


def _run(**kwargs):
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        global_batch_size=1,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    captured = {}
    model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            opt_param_scheduler=None,
            config=SimpleNamespace(),
            forward_backward_func=lambda **kw: captured.update(kw) or [],
            iteration=0,
            **kwargs,
        )
    return captured


def test_train_step_forwards_schedule_plumbing():
    p2p, pg = object(), object()
    captured = _run(p2p_communicator=p2p, schedule_pg_collection=pg)
    assert captured["p2p_communicator"] is p2p and captured["pg_collection"] is pg


def test_train_step_defaults_to_none():
    captured = _run()
    assert captured["p2p_communicator"] is None and captured["pg_collection"] is None


class _FakeTorchFSDP:
    def no_sync(self):
        """Stand in for the FSDP2 no_sync context factory."""


@pytest.mark.parametrize("num_model_chunks", [1, 2])
def test_configure_torch_fsdp2_no_sync(monkeypatch, num_model_chunks):
    """Install FSDP2 no_sync callbacks even without overlap_grad_reduce."""
    monkeypatch.setattr(training_mod, "HAVE_FSDP2", True)
    monkeypatch.setattr(training_mod, "torch_FSDP", _FakeTorchFSDP)
    model = [_FakeTorchFSDP() for _ in range(num_model_chunks)]
    config = SimpleNamespace(no_sync_func=None)

    assert training_mod._configure_torch_fsdp2_no_sync(model, config)

    no_sync_funcs = (
        config.no_sync_func if isinstance(config.no_sync_func, list) else [config.no_sync_func]
    )
    assert len(no_sync_funcs) == num_model_chunks
    assert all(
        callback.__self__ is model_chunk for callback, model_chunk in zip(no_sync_funcs, model)
    )


def test_configure_torch_fsdp2_no_sync_rejects_mixed_wrappers(monkeypatch):
    """Reject pipeline chunks that do not all use the Torch FSDP2 wrapper."""
    monkeypatch.setattr(training_mod, "HAVE_FSDP2", True)
    monkeypatch.setattr(training_mod, "torch_FSDP", _FakeTorchFSDP)
    config = SimpleNamespace(no_sync_func=None)

    with pytest.raises(AssertionError, match="all model chunks"):
        training_mod._configure_torch_fsdp2_no_sync([_FakeTorchFSDP(), object()], config)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", None),
        ("AUTO", None),
        ("true", True),
        ("TRUE", True),
        ("false", False),
        ("FALSE", False),
        ("2", 2),
        ("8", 8),
    ],
)
def test_parse_torch_fsdp2_reshard_after_forward(value, expected):
    """Parse every supported automatic, boolean, and integer policy spelling."""
    parsed = arguments_mod._parse_torch_fsdp2_reshard_after_forward(value)

    assert parsed == expected
    assert type(parsed) is type(expected)


@pytest.mark.parametrize("value", ["none", "yes", "0", "1", "-2", "2.5"])
def test_parse_torch_fsdp2_reshard_after_forward_rejects_invalid_values(value):
    """Reject ambiguous policy names and invalid partial-reshard world sizes."""
    with pytest.raises(argparse.ArgumentTypeError):
        arguments_mod._parse_torch_fsdp2_reshard_after_forward(value)


@pytest.mark.parametrize(
    ("cli_args", "expected"),
    [
        ([], None),
        (["--torch-fsdp2-reshard-after-forward", "auto"], None),
        (["--torch-fsdp2-reshard-after-forward", "true"], True),
        (["--torch-fsdp2-reshard-after-forward", "false"], False),
        (["--torch-fsdp2-reshard-after-forward", "4"], 4),
        (["--torch-fsdp2-no-reshard-after-forward"], False),
    ],
)
def test_torch_fsdp2_reshard_cli_preserves_legacy_flag(cli_args, expected):
    """Default to auto while preserving the legacy no-reshard flag."""
    parser = argparse.ArgumentParser()
    arguments_mod._add_distributed_args(parser)

    parsed = parser.parse_args(cli_args).torch_fsdp2_reshard_after_forward

    assert parsed == expected
    assert type(parsed) is type(expected)


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--torch-fsdp2-reshard-after-forward", "auto", "--torch-fsdp2-no-reshard-after-forward"],
        ["--torch-fsdp2-no-reshard-after-forward", "--torch-fsdp2-reshard-after-forward", "auto"],
    ],
)
def test_torch_fsdp2_reshard_cli_rejects_conflicting_flags(cli_args):
    """Do not let argument order silently choose between new and legacy policies."""
    parser = argparse.ArgumentParser()
    arguments_mod._add_distributed_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(cli_args)


@pytest.mark.parametrize("reduce_scatter_unused_params", [False, True])
@pytest.mark.parametrize("clone_output_views", [False, True])
@pytest.mark.parametrize("reshard_after_forward", [None, True, False, 2])
def test_get_megatron_ddp_config_forwards_torch_fsdp2_options(
    reduce_scatter_unused_params, clone_output_views, reshard_after_forward
):
    """Forward all Torch FSDP2-specific CLI options into its config."""
    args = SimpleNamespace(
        use_torch_fsdp2=True,
        torch_fsdp2_reshard_after_forward=reshard_after_forward,
        torch_fsdp2_reduce_scatter_unused_params=reduce_scatter_unused_params,
        torch_fsdp2_clone_output_views=clone_output_views,
    )

    config = training_mod.get_megatron_ddp_config(args)

    assert config.reshard_after_forward == reshard_after_forward
    assert type(config.reshard_after_forward) is type(reshard_after_forward)
    assert config.reduce_scatter_unused_params is reduce_scatter_unused_params
    assert config.clone_output_views is clone_output_views


def test_get_megatron_ddp_config_defaults_torch_fsdp2_to_auto_reshard():
    """Use the automatic policy for programmatic argument namespaces without the new field."""
    config = training_mod.get_megatron_ddp_config(SimpleNamespace(use_torch_fsdp2=True))

    assert config.reshard_after_forward is None
