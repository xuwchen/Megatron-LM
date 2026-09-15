# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep both convergence arms on the same model, packing, and optimizer controls."""

from types import SimpleNamespace

import pytest

from examples.multimodal_dev.data.nemotron.run_convergence import build_command


def launch_args(tmp_path, dataset):
    """Supply only the arm-specific inputs of the portable comparison entry."""
    return SimpleNamespace(
        dataset=dataset,
        selection=tmp_path / 'selection.json' if dataset == 'nemotron' else None,
        iterations=500,
        output_dir=tmp_path / dataset,
        run_name=dataset,
        wandb_project='nemotron-image-dataset',
        wandb_entity=None,
        group='matched-controls',
    )


def command_args(command):
    """Parse emitted Megatron flags after the torchrun module entry."""
    command = command[command.index('examples.multimodal_dev.pretrain_multimodal') + 1 :]
    result = {}
    for value in command:
        if value.startswith('--'):
            flag = value
            result[flag] = []
        else:
            result[flag].append(value)
    return result


def test_real_and_mock_share_the_verified_pr57_controls(tmp_path):
    """Only input selection and run identity may differ between the two arms."""
    real, real_env = build_command(launch_args(tmp_path, 'nemotron'))
    mock, mock_env = build_command(launch_args(tmp_path, 'mdp_mock'))
    assert real_env == mock_env
    assert real[:8] == mock[:8]
    real, mock = command_args(real), command_args(mock)
    assert real.pop('--data-path') == [str(tmp_path / 'selection.json')]
    for key in ['--dataset-provider', '--tensorboard-dir', '--wandb-save-dir', '--wandb-exp-name']:
        assert real.pop(key) != mock.pop(key)
    assert real == mock
    for key, expected in {
        '--model-variant': '397b_a17b_light',
        '--num-layers': '8',
        '--hidden-size': '4096',
        '--num-experts': '32',
        '--moe-router-topk': '10',
        '--vision-num-layers': '8',
        '--mtp-num-layers': '1',
        '--tensor-model-parallel-size': '1',
        '--pipeline-model-parallel-size': '2',
        '--expert-model-parallel-size': '2',
        '--context-parallel-size': '1',
        '--train-iters': '500',
        '--seq-length': '32768',
        '--max-seqlen-per-dp-cp-rank': '32768',
        '--thd-max-packed-sequences': '64',
        '--mdp-encoder-max-payload-rows': '8192',
        '--encoder-recompute-granularity': 'whole',
        '--micro-batch-size': '1',
        '--global-batch-size': '8',
    }.items():
        assert real[key] == [expected]
    assert real['--mdp-greedy-packing'] == real['--thd-static-packing'] == []
    assert '--moe-router-force-load-balancing' not in real


@pytest.mark.parametrize('dataset', ['nemotron', 'mdp_mock'])
def test_selection_must_match_the_requested_data_provider(tmp_path, dataset):
    """Reject an omitted real selection or one accidentally attached to mock."""
    args = launch_args(tmp_path, dataset)
    args.selection = None if dataset == 'nemotron' else tmp_path / 'selection.json'
    with pytest.raises(ValueError, match='selection'):
        build_command(args)
