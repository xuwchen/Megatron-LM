# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run one arm of the PR57 A17B-light 32K MDP convergence comparison."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Resolve shared controls and add only dataset and run-output arguments."""
    if args.iterations < 1:
        raise ValueError('Iterations must be positive')
    if args.dataset == 'nemotron' and args.selection is None:
        raise ValueError('Nemotron requires --selection from prepare_training.py')
    if args.dataset == 'mdp_mock' and args.selection is not None:
        raise ValueError('Mock does not consume a real-data selection')
    recipe = json.loads(Path(__file__).with_name('convergence_32k.json').read_text())
    controls = recipe['args']
    controls.update(
        dataset_provider=args.dataset,
        train_iters=args.iterations,
        tensorboard_dir=str(args.output_dir.resolve() / 'tensorboard'),
        wandb_save_dir=str(args.output_dir.resolve() / 'wandb'),
        wandb_project=args.wandb_project,
        wandb_exp_name=args.run_name,
    )
    if args.wandb_entity:
        controls['wandb_entity'] = args.wandb_entity
    if args.selection is not None:
        controls['data_path'] = [str(args.selection.resolve())]
    command = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc-per-node',
        str(recipe['nproc_per_node']),
        '-m',
        'examples.multimodal_dev.pretrain_multimodal',
    ]
    for name, value in controls.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.append(flag)
            command.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    return command, {
        **recipe['env'],
        'WANDB_RUN_GROUP': args.group,
        'WANDB_JOB_TYPE': 'convergence',
    }


def main() -> None:
    """Render or execute one matched training arm inside a prepared GPU container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['nemotron', 'mdp_mock'], required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=500, help='Use 5 for a smoke run')
    parser.add_argument('--wandb-project', default='nemotron-image-dataset')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--group', default='pr57-a17b-light-pp2-ep2-32k')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command, environment = build_command(args)
    sys.stdout.write(
        json.dumps({'environment': environment, 'command': shlex.join(command)}, indent=2) + '\n'
    )
    if args.dry_run:
        return
    # The provider validates the selection, its provenance and its exact budget.
    # Do not permit a caller to accidentally mix files from two experiments.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'launch.json').write_text(
        json.dumps({'environment': environment, 'command': command}, indent=2) + '\n'
    )
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, **environment},
        check=True,
    )


if __name__ == '__main__':
    main()
