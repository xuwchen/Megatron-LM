# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Export complete W&B scalar histories and a real/mock learning-curve figure."""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)


def finite(value: Any) -> bool:
    """Return whether a scalar metric is a finite real number."""
    return isinstance(value, (int, float)) and math.isfinite(value)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize training scalars and separately logged evaluation points."""
    losses = [r['lm loss'] for r in rows if finite(r.get('lm loss'))]
    grads = [r['grad-norm'] for r in rows if finite(r.get('grad-norm'))]
    validation = [
        {'iteration': r['_step'], 'loss': r['lm loss validation']}
        for r in rows
        if finite(r.get('lm loss validation'))
    ]
    return {
        'logged_training_steps': len(losses),
        'last_training_iteration': max(
            (r['_step'] for r in rows if finite(r.get('lm loss'))), default=None
        ),
        'first_loss': losses[0] if losses else None,
        'final_loss': losses[-1] if losses else None,
        'first_20_mean_loss': statistics.mean(losses[:20]) if losses else None,
        'last_20_mean_loss': statistics.mean(losses[-20:]) if losses else None,
        'max_grad_norm': max(grads, default=None),
        'validation': validation,
    }


def verify_history(rows: list[dict[str, Any]], expected_iterations: int) -> dict[str, Any]:
    """Require an uninterrupted finite training history with parameter updates."""
    if expected_iterations < 1:
        raise ValueError('Expected training iteration count must be positive')
    train = [row for row in rows if 'lm loss' in row]
    steps = [row.get('_step') for row in train]
    if len(steps) != expected_iterations or set(steps) != set(range(1, expected_iterations + 1)):
        raise ValueError('Training history has missing, duplicate or unexpected iterations')
    for row in train:
        if not all(finite(row.get(key)) for key in ('lm loss', 'grad-norm', 'params-norm')):
            raise ValueError(f'Nonfinite or absent training metric at iteration {row.get("_step")}')
    if expected_iterations > 1 and len({row['params-norm'] for row in train}) < 2:
        raise ValueError('Parameter norms do not show an optimizer update')
    return {
        'iterations': expected_iterations,
        'finite_loss_and_gradients': True,
        'parameter_norm_changed': expected_iterations > 1,
    }


def main() -> None:
    """Read full W&B histories and export a checked comparison plot locally."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-run', required=True, help='entity/project/run_id')
    parser.add_argument('--mock-run', required=True, help='entity/project/run_id')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--expect-iterations', type=int, help='Require finished runs with this complete history'
    )
    parser.add_argument('--real-label', default='CLEVR 2')
    parser.add_argument('--model-label', default='Qwen3.5-VL proxy')
    args = parser.parse_args()
    import wandb

    api = wandb.Api(timeout=60)
    report = {
        'scope': (
            f'Random-initialized {args.model_label}, 32K static THD, '
            f'real {args.real_label} vs native MDP mock'
        ),
        'runs': {},
    }
    selected = {
        '_step',
        'lm loss',
        'lm loss validation',
        'grad-norm',
        'params-norm',
        'learning-rate',
        'iteration-time',
        'samples vs steps',
        'num-zeros',
        'batch-size',
        'skipped-train-samples',
    }
    for arm, path in [(args.real_label, args.real_run), ('Mock', args.mock_run)]:
        run = api.run(path)
        # Do not request multiple keys: scan_history would then omit rows missing any key.
        rows = [
            {k: v for k, v in row.items() if k in selected}
            for row in run.scan_history(page_size=1000)
        ]
        rows.sort(key=lambda r: r.get('_step', -1))
        summary = summarize(rows)
        if args.expect_iterations is not None:
            if run.state != 'finished':
                raise ValueError(f'{arm} run is not finished: {run.state}')
            summary['completion_checks'] = verify_history(rows, args.expect_iterations)
        if arm == 'Mock' and run.state == 'finished':
            # Native MDP mock supplies a test split; vendor reuses the validation key.
            final_iteration = int(run.config['train_iters'])
            for point in summary['validation']:
                if point['iteration'] == final_iteration:
                    point['split'] = 'test (native vendor reuses validation metric name)'
        report['runs'][arm] = {
            'url': run.url,
            'state': run.state,
            'name': run.name,
            'summary': summary,
            'history': rows,
        }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'wandb-history.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {args.real_label: '#166a96', 'Mock': '#cd702b'}
    for arm, record in report['runs'].items():
        rows = record['history']
        train = [(r['_step'], r['lm loss']) for r in rows if finite(r.get('lm loss'))]
        if train:
            x, y = zip(*train)
            axes[0].plot(x, y, color=colors[arm], alpha=0.18, linewidth=0.7)
            smooth = [statistics.mean(y[max(0, i - 19) : i + 1]) for i in range(len(y))]
            axes[0].plot(x, smooth, color=colors[arm], label=arm)
        for ax, key in [(axes[1], 'lm loss validation'), (axes[2], 'grad-norm')]:
            values = [(r['_step'], r[key]) for r in rows if finite(r.get(key))]
            if arm == 'Mock' and record['state'] == 'finished' and ax is axes[1]:
                test_steps = {
                    p['iteration'] for p in record['summary']['validation'] if 'split' in p
                }
                test_values = [(x, y) for x, y in values if x in test_steps]
                values = [(x, y) for x, y in values if x not in test_steps]
                if test_values:
                    x, y = zip(*test_values)
                    ax.scatter(
                        x, y, color=colors[arm], marker='x', label='Mock test (final)', zorder=3
                    )
            if values:
                x, y = zip(*values)
                ax.plot(
                    x,
                    y,
                    color=colors[arm],
                    label=arm,
                    marker='o' if ax is axes[1] else None,
                    markersize=4,
                    linewidth=1.2,
                )
    for ax, title, ylabel in zip(
        axes,
        ['Training loss (20-step trailing mean)', 'Validation loss', 'Gradient norm'],
        [
            'Cross-entropy per target token',
            'Cross-entropy per target token',
            'L2 norm before clipping',
        ],
    ):
        ax.set(title=title, xlabel='Optimizer iteration', ylabel=ylabel)
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(frameon=False)
    fig.suptitle(f'{args.model_label} | 32K static THD | random initialization', fontsize=14)
    fig.savefig(out / 'convergence.png', dpi=180)
    fig.savefig(out / 'convergence.pdf')
    plt.close(fig)
    _LOG.info(
        json.dumps(
            {
                arm: {k: v for k, v in record.items() if k != 'history'}
                for arm, record in report['runs'].items()
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
