# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A convergence plot must not certify missing or nonfinite optimizer steps."""
import pytest

from examples.multimodal_dev.data.nemotron.report_convergence import verify_history


def history():
    """Provide a complete three-step history with observable parameter updates."""
    return [
        {'_step': i, 'lm loss': 13 - i, 'grad-norm': 0.5, 'params-norm': 10 + i * 0.1}
        for i in range(1, 4)
    ]


def test_complete_training_history_ignores_separate_evaluation_rows():
    """Keep evaluation rows separate from the optimizer-step completeness check."""
    rows = history() + [{'_step': 3, 'lm loss validation': 9}]
    assert verify_history(rows, 3) == {
        'iterations': 3,
        'finite_loss_and_gradients': True,
        'parameter_norm_changed': True,
    }


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'loss_nan', 'grad_nan', 'no_update'])
def test_invalid_history_is_not_reported_as_completed(problem):
    """Refuse incomplete, nonfinite or non-updating histories."""
    rows = history()
    if problem == 'missing':
        rows.pop()
    if problem == 'duplicate':
        rows[2]['_step'] = 2
    if problem == 'loss_nan':
        rows[1]['lm loss'] = float('nan')
    if problem == 'grad_nan':
        rows[1]['grad-norm'] = float('nan')
    if problem == 'no_update':
        for row in rows:
            row['params-norm'] = 10
    with pytest.raises(ValueError):
        verify_history(rows, 3)


def test_empty_run_cannot_certify_training():
    """Reject a zero-iteration completion claim."""
    with pytest.raises(ValueError, match='positive'):
        verify_history([], 0)
