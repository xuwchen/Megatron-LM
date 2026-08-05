# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import contextmanager

import pytest
import torch

import megatron.core.tensor_parallel.random as random_module
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CudaRNGStatesTracker,
    checkpoint,
    convert_cuda_rng_state,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from tests.unit_tests.test_utilities import Utils


def test_cuda_rng_states_tracker():
    rng_tracker = CudaRNGStatesTracker()
    rng_tracker.set_states({"state1": 1234})
    assert rng_tracker.get_states()["state1"] == 1234
    rng_tracker.reset()
    assert rng_tracker.get_states() == {}
    seed = 1111
    rng_tracker.add("state2", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state3", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state2", 111)
    assert rng_tracker.get_states()['state2'] is not None
    with pytest.raises(Exception):
        assert ()

    rng_tracker.fork("state2")
    torch.cuda.manual_seed(seed)
    rng_state = torch.cuda.get_rng_state()
    assert torch.equal(rng_tracker.get_states()['state2'], rng_state)


@pytest.mark.parametrize("use_cudagraphable_rng", [True, False])
def test_double_fork_cuda_rng_states_tracker(use_cudagraphable_rng):
    rng_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=use_cudagraphable_rng)
    rng_tracker.add("state1", 1234)
    rng_tracker.add("state2", 5678)
    randn_double_fork_1 = []
    randn_double_fork_2 = []
    with rng_tracker.fork("state1"):
        randn_double_fork_1.append(torch.randn(10, device="cuda"))
        with rng_tracker.fork("state2"):
            randn_double_fork_2.append(torch.randn(10, device="cuda"))
            with rng_tracker.fork("state1"):
                randn_double_fork_1.append(torch.randn(10, device="cuda"))
            randn_double_fork_2.append(torch.randn(10, device="cuda"))
        randn_double_fork_1.append(torch.randn(10, device="cuda"))
    if use_cudagraphable_rng:
        double_fork_state1 = rng_tracker.get_states()["state1"].get_state()
        double_fork_state2 = rng_tracker.get_states()["state2"].get_state()
    else:
        double_fork_state1 = rng_tracker.get_states()["state1"]
        double_fork_state2 = rng_tracker.get_states()["state2"]

    rng_tracker.reset()
    rng_tracker.add("state1", 1234)
    rng_tracker.add("state2", 5678)
    randn_single_fork_1 = []
    randn_single_fork_2 = []
    with rng_tracker.fork("state1"):
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
    with rng_tracker.fork("state2"):
        randn_single_fork_2.append(torch.randn(10, device="cuda"))
        randn_single_fork_2.append(torch.randn(10, device="cuda"))
    if use_cudagraphable_rng:
        single_fork_state1 = rng_tracker.get_states()["state1"].get_state()
        single_fork_state2 = rng_tracker.get_states()["state2"].get_state()
    else:
        single_fork_state1 = rng_tracker.get_states()["state1"]
        single_fork_state2 = rng_tracker.get_states()["state2"]

    assert torch.equal(randn_double_fork_1[0], randn_single_fork_1[0])
    assert torch.equal(randn_double_fork_1[1], randn_single_fork_1[1])
    assert torch.equal(randn_double_fork_1[2], randn_single_fork_1[2])
    assert torch.equal(randn_double_fork_2[0], randn_single_fork_2[0])
    assert torch.equal(randn_double_fork_2[1], randn_single_fork_2[1])
    assert torch.equal(double_fork_state1, single_fork_state1)
    assert torch.equal(double_fork_state2, single_fork_state2)


def test_convert_cuda_rng_state():
    ## Get the default rng state
    torch.cuda.manual_seed(999)
    randn = torch.randn(10, device="cuda")
    rng_state = torch.cuda.get_rng_state()

    try:
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
    except ImportError:
        TECudaRNGStatesTracker = None

    ## from non-graphable RNG to graphable RNG
    # get state from non-graphable RNG
    tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
    tracker.add("state1", 123)
    for i in range(3):
        with tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
    state = convert_cuda_rng_state(tracker.states_["state1"], to_graphable=True)
    rand_tensors = []
    for i in range(3):
        with tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
            rand_tensors.append(randn)

    # set state to local graph RNG
    cudagraphable_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=True)
    cudagraphable_tracker.set_states({"state1": state.clone_state()})
    for i in range(3):
        with cudagraphable_tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
            assert torch.equal(randn, rand_tensors[i])

    # set state to TE RNG
    if TECudaRNGStatesTracker is not None:
        te_tracker = TECudaRNGStatesTracker()
        te_tracker.set_states({"state1": state})
        for i in range(3):
            with te_tracker.fork("state1"):
                randn = torch.randn(10, device="cuda")
                assert torch.equal(randn, rand_tensors[i])

    ## from graphable RNG to non-graphable RNG
    # get state from graphable RNG
    cudagraphable_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=True)
    cudagraphable_tracker.add("state2", 123)
    for i in range(3):
        with cudagraphable_tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
    state = convert_cuda_rng_state(cudagraphable_tracker.states_["state2"], to_graphable=False)
    rand_tensors = []
    for i in range(3):
        with cudagraphable_tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
            rand_tensors.append(randn)

    # set state to non-graphable RNG
    tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
    tracker.set_states({"state2": state})
    for i in range(3):
        with tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
            assert torch.equal(randn, rand_tensors[i])

    ## from TE RNG to non-graphable RNG
    if TECudaRNGStatesTracker is not None:
        # get state from TE RNG
        cudagraphable_tracker = TECudaRNGStatesTracker()
        cudagraphable_tracker.add("state3", 123)
        for i in range(3):
            with cudagraphable_tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
        state = convert_cuda_rng_state(cudagraphable_tracker.states_["state3"], to_graphable=False)
        rand_tensors = []
        for i in range(3):
            with cudagraphable_tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
                rand_tensors.append(randn)

        # set state to non-graphable RNG
        tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
        tracker.set_states({"state3": state})
        for i in range(3):
            with tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
                assert torch.equal(randn, rand_tensors[i])

    ## After all tests, check if the default rng state is still the same.
    rng_state_final = torch.cuda.get_rng_state()
    assert torch.equal(rng_state, rng_state_final)


def test_model_parallel_cuda_manual_seed():
    Utils.initialize_model_parallel(4, 2)
    model_parallel_cuda_manual_seed(0, force_reset_rng=True)
    rng_tracker = get_cuda_rng_tracker()
    assert rng_tracker.get_states()['model-parallel-rng'] is not None
    Utils.destroy_model_parallel()


def test_checkpoint():
    def test_forward(*input):
        return input[0] + input[1]

    assert torch.equal(
        torch.ones(16) * 3, checkpoint(test_forward, None, torch.ones(16), torch.ones(16) * 2)
    )

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    input1 = torch.ones((4, 4)).cuda()
    input1.requires_grad_(True)
    input2 = torch.ones((4, 4)).cuda() * 2
    output = checkpoint(test_forward, True, input1, input2)

    assert torch.equal(output, torch.ones((4, 4)).cuda() * 3)
    assert input1.data.shape == (8,)

    output.sum().backward()
    assert input1.grad is not None
    assert torch.equal(input1.grad, torch.ones((4, 4)).cuda())

    Utils.destroy_model_parallel()


def test_checkpoint_without_output():
    def normal_forward(input):
        x = torch.nn.functional.gelu(input)
        y = x * input
        return y

    def checkpoint_forward(input):
        checkpoint = CheckpointWithoutOutput()
        x = checkpoint.checkpoint(torch.nn.functional.gelu, input)
        y = x * input
        checkpoint.discard_output_and_register_recompute(y)
        return y

    Utils.initialize_model_parallel()

    input1 = torch.ones((4, 4))
    input1.requires_grad_(True)
    output1 = normal_forward(input1)
    input2 = torch.ones((4, 4))
    input2.requires_grad_(True)
    output2 = checkpoint_forward(input2)
    assert torch.equal(output1, output2)

    output1.backward(torch.ones((4, 4)), retain_graph=True)
    output2.backward(torch.ones((4, 4)), retain_graph=True)
    assert torch.equal(input1.grad, input2.grad)


def test_checkpoint_te_recompute_marker_is_independent_of_fp8(monkeypatch):
    phases = []

    @contextmanager
    def activation_recompute_context(*, activation_recompute, recompute_phase):
        assert activation_recompute
        phases.append(recompute_phase)
        yield

    monkeypatch.setattr(random_module, "HAVE_TE", True)
    monkeypatch.setattr(
        random_module, "activation_recompute_forward", activation_recompute_context, raising=False
    )
    monkeypatch.setattr(random_module, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(random_module, "_set_all_rng_states", lambda *args: None)

    @contextmanager
    def fork_rng():
        yield

    monkeypatch.setattr(random_module, "_fork_rng", fork_rng)

    input_tensor = torch.randn(4, 4, requires_grad=True)
    output = checkpoint(torch.nn.functional.gelu, False, input_tensor, te_activation_recompute=True)
    output.sum().backward()

    assert phases == [False, True]
    assert input_tensor.grad is not None


def test_checkpoint_restores_state_after_forward_exception(monkeypatch):
    monkeypatch.setattr(random_module, "IS_CHECKPOINTING", False)
    monkeypatch.setattr(random_module, "_get_all_rng_states", lambda: ())

    def failing_forward(input_tensor):
        assert random_module.is_checkpointing()
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        checkpoint(failing_forward, False, torch.randn(4, requires_grad=True))

    assert not random_module.is_checkpointing()


def test_checkpoint_restores_state_after_backward_replay_exception(monkeypatch):
    monkeypatch.setattr(random_module, "IS_CHECKPOINTING", False)
    monkeypatch.setattr(random_module, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(random_module, "_set_all_rng_states", lambda *args: None)

    @contextmanager
    def fork_rng():
        yield

    monkeypatch.setattr(random_module, "_fork_rng", fork_rng)
    calls = 0

    def fail_during_replay(input_tensor):
        nonlocal calls
        calls += 1
        assert random_module.is_checkpointing()
        if calls == 2:
            raise RuntimeError("backward replay failed")
        return input_tensor.square()

    input_tensor = torch.randn(4, requires_grad=True)
    output = checkpoint(fail_during_replay, False, input_tensor)
    assert not random_module.is_checkpointing()

    with pytest.raises(RuntimeError, match="backward replay failed"):
        output.sum().backward()

    assert calls == 2
    assert not random_module.is_checkpointing()


def test_checkpoint_without_output_te_recompute_marker_is_independent_of_fp8(monkeypatch):
    phases = []

    @contextmanager
    def activation_recompute_context(*, activation_recompute, recompute_phase):
        assert activation_recompute
        phases.append(recompute_phase)
        yield

    monkeypatch.setattr(random_module, "HAVE_TE", True)
    monkeypatch.setattr(
        random_module, "activation_recompute_forward", activation_recompute_context, raising=False
    )
    monkeypatch.setattr(random_module, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(random_module, "_set_all_rng_states", lambda *args: None)

    @contextmanager
    def fork_rng():
        yield

    monkeypatch.setattr(random_module, "_fork_rng", fork_rng)
    monkeypatch.setattr(random_module, "_get_share_storage", lambda: lambda *_: None)

    input_tensor = torch.randn(4, 4, requires_grad=True)
    checkpoint = CheckpointWithoutOutput(te_activation_recompute=True)
    assert not checkpoint.fp8

    checkpoint.checkpoint(torch.nn.functional.gelu, input_tensor)
    checkpoint._recompute(None)

    assert phases == [False, True]


class _ViewSavingLinear(torch.autograd.Function):
    """Saves view tensors in forward to mimic TE GroupedLinear-style backward inputs."""

    @staticmethod
    def forward(ctx, inp, weight):
        inp_2d = inp.reshape(-1, inp.shape[-1])
        inputmats = torch.tensor_split(inp_2d, 2, dim=0)
        ctx.save_for_backward(*inputmats, weight)
        ctx.input_shape = inp.shape
        out_2d = inp_2d.matmul(weight.t())
        return out_2d.reshape(*inp.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        *inputmats, weight = ctx.saved_tensors
        for inputmat in inputmats:
            if inputmat.numel() > 0 and inputmat.untyped_storage().size() == 0:
                raise RuntimeError("Saved view tensor points to an empty storage.")

        inp_2d = torch.cat(inputmats, dim=0)
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_input_2d = grad_output_2d.matmul(weight)
        grad_weight = grad_output_2d.t().matmul(inp_2d)
        grad_input = grad_input_2d.reshape(ctx.input_shape)
        return grad_input, grad_weight


def test_checkpoint_without_output_view_sharing_regression():
    def normal_forward(input_, weight):
        x = torch.nn.functional.gelu(input_)
        return _ViewSavingLinear.apply(x, weight)

    def checkpoint_forward(input_, weight):
        checkpoint = CheckpointWithoutOutput()
        x = checkpoint.checkpoint(torch.nn.functional.gelu, input_)
        y = _ViewSavingLinear.apply(x, weight)
        checkpoint.discard_output_and_register_recompute(y)
        return y

    Utils.initialize_model_parallel()
    try:
        input1 = torch.randn((3, 2, 8), requires_grad=True)
        weight1 = torch.randn((6, 8), requires_grad=True)

        input2 = input1.detach().clone().requires_grad_(True)
        weight2 = weight1.detach().clone().requires_grad_(True)

        output1 = normal_forward(input1, weight1)
        output2 = checkpoint_forward(input2, weight2)
        assert torch.allclose(output1, output2)

        grad = torch.randn_like(output1)
        output1.backward(grad, retain_graph=True)
        output2.backward(grad, retain_graph=True)
        assert torch.allclose(input1.grad, input2.grad)
        assert torch.allclose(weight1.grad, weight2.grad)
    finally:
        Utils.destroy_model_parallel()
