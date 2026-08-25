"""Standalone (non-MDP, non-distributed-launch) numerical equivalence + training-state
regression test for the "pad vision FFN ffn_hidden_size 4304 -> 4320 with zero
padding" MXFP8 approach (Approach B).

Run directly with `python3 scripts/test_mxfp8_ffn_padding.py` inside a container with
torch + transformer_engine installed, on a single GPU (torch.distributed init with
world_size=1 is required because MCore's TE wrappers assume parallel_state is
initialized).
"""

import os

import torch
import torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

dist.init_process_group(backend="nccl", rank=0, world_size=1)
torch.cuda.set_device(0)

from megatron.core import parallel_state  # noqa: E402

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)

from megatron.core.models.vision.vit_layer_specs import (  # noqa: E402
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.mlp import MLP  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402


def build_config(ffn_hidden_size: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=1152,
        num_attention_heads=16,
        kv_channels=72,
        ffn_hidden_size=ffn_hidden_size,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-6,
        normalization="LayerNorm",
        gated_linear_unit=False,
        activation_func=lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        bias_activation_fusion=False,
        apply_query_key_layer_scaling=False,
        apply_rope_fusion=False,
        add_bias_linear=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        perform_initialization=True,
        pipeline_dtype=torch.bfloat16,
    )


def build_mlp(config: TransformerConfig) -> MLP:
    spec = get_vit_layer_with_transformer_engine_spec()
    mlp_partial = spec.submodules.mlp
    mlp_submodules = mlp_partial.keywords["submodules"]
    mlp = MLP(config, submodules=mlp_submodules)
    return mlp.cuda()


def main():
    torch.manual_seed(1234)

    HIDDEN = 1152
    REAL_FFN = 4304
    PAD_FFN = 4320
    S, B = 37, 2  # deliberately not a multiple of anything special

    cfg_a = build_config(REAL_FFN)
    mlp_a = build_mlp(cfg_a)

    cfg_b = build_config(PAD_FFN)
    mlp_b = build_mlp(cfg_b)

    with torch.no_grad():
        # TE column/row parallel linear .weight is [out_features, in_features],
        # standard nn.Linear convention (TP=1 here so no sharding to worry about).
        w1a = mlp_a.linear_fc1.weight.data  # [4304, 1152]
        w1b = mlp_b.linear_fc1.weight.data  # [4320, 1152]
        assert w1a.shape == (REAL_FFN, HIDDEN), w1a.shape
        assert w1b.shape == (PAD_FFN, HIDDEN), w1b.shape
        w1b.zero_()
        w1b[:REAL_FFN, :].copy_(w1a)

        b1a = mlp_a.linear_fc1.bias.data  # [4304]
        b1b = mlp_b.linear_fc1.bias.data  # [4320]
        b1b.zero_()
        b1b[:REAL_FFN].copy_(b1a)

        w2a = mlp_a.linear_fc2.weight.data  # [1152, 4304]
        w2b = mlp_b.linear_fc2.weight.data  # [1152, 4320]
        assert w2a.shape == (HIDDEN, REAL_FFN), w2a.shape
        assert w2b.shape == (HIDDEN, PAD_FFN), w2b.shape
        w2b.zero_()
        w2b[:, :REAL_FFN].copy_(w2a)

        if mlp_a.linear_fc2.bias is not None:
            mlp_b.linear_fc2.bias.data.copy_(mlp_a.linear_fc2.bias.data)

    x = torch.randn(S, B, HIDDEN, device="cuda", dtype=torch.bfloat16)
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)

    # ---- Step 1: forward numerical equivalence (no fp8) ----
    out_a, _ = mlp_a(xa)
    out_b, _ = mlp_b(xb)
    equal = torch.equal(out_a, out_b)
    max_abs_diff = (out_a - out_b).abs().max().item()
    print(f"[step1] bf16 forward exact equal={equal} max_abs_diff={max_abs_diff}")
    assert equal, f"Step 1 FAILED: padded MLP output diverges from unpadded, max_abs_diff={max_abs_diff}"

    # ---- Step 2: backward -- padding gradients must be exactly zero ----
    loss_a = out_a.float().pow(2).sum()
    loss_b = out_b.float().pow(2).sum()
    loss_a.backward()
    loss_b.backward()

    g_w1_extra = mlp_b.linear_fc1.weight.grad[REAL_FFN:, :]
    g_b1_extra = mlp_b.linear_fc1.bias.grad[REAL_FFN:]
    g_w2_extra = mlp_b.linear_fc2.weight.grad[:, REAL_FFN:]
    print(
        f"[step2] grad on padding: "
        f"fc1.weight max_abs={g_w1_extra.abs().max().item()} "
        f"fc1.bias max_abs={g_b1_extra.abs().max().item()} "
        f"fc2.weight max_abs={g_w2_extra.abs().max().item()}"
    )
    assert torch.equal(g_w1_extra, torch.zeros_like(g_w1_extra)), "fc1 padding weight grad nonzero"
    assert torch.equal(g_b1_extra, torch.zeros_like(g_b1_extra)), "fc1 padding bias grad nonzero"
    assert torch.equal(g_w2_extra, torch.zeros_like(g_w2_extra)), "fc2 padding weight grad nonzero"

    real_grad_a = mlp_a.linear_fc1.weight.grad
    real_grad_b = mlp_b.linear_fc1.weight.grad[:REAL_FFN, :]
    real_equal = torch.equal(real_grad_a, real_grad_b)
    print(f"[step2] real-channel fc1 weight grad exact equal={real_equal}")
    assert real_equal

    # ---- Step 3: repeated optimizer steps -- padding must stay exactly at init ----
    opt_a = torch.optim.Adam(mlp_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(mlp_b.parameters(), lr=1e-3)
    opt_a.zero_grad()
    opt_b.zero_grad()

    all_zero_throughout = True
    for it in range(20):
        xa = torch.randn(S, B, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        xb = xa.detach().clone().requires_grad_(True)
        out_a, _ = mlp_a(xa)
        out_b, _ = mlp_b(xb)
        loss_a = out_a.float().pow(2).mean()
        loss_b = out_b.float().pow(2).mean()
        opt_a.zero_grad()
        opt_b.zero_grad()
        loss_a.backward()
        loss_b.backward()
        opt_a.step()
        opt_b.step()

        w1_extra = mlp_b.linear_fc1.weight.data[REAL_FFN:, :]
        w2_extra = mlp_b.linear_fc2.weight.data[:, REAL_FFN:]
        if w1_extra.abs().max().item() != 0.0 or w2_extra.abs().max().item() != 0.0:
            all_zero_throughout = False
            print(f"[step3] iter {it}: padding drifted from zero! "
                  f"fc1_extra_max={w1_extra.abs().max().item()} fc2_extra_max={w2_extra.abs().max().item()}")

    print(f"[step3] padding stayed exactly zero for all 20 optimizer steps: {all_zero_throughout}")
    assert all_zero_throughout, "Step 3 FAILED: padding channels drifted away from zero during training"

    # ---- Step 4: real-channel output still matches after 20 training steps ----
    x_final = torch.randn(S, B, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out_a_final, _ = mlp_a(x_final.clone())
    out_b_final, _ = mlp_b(x_final.clone())
    final_equal = torch.equal(out_a_final, out_b_final)
    print(f"[step4] after 20 training steps, forward outputs still exact-equal: {final_equal}")
    assert final_equal

    print("\nALL CHECKS PASSED (bf16 path): zero-padding ffn_hidden_size 4304->4320 is a "
          "mathematically exact, self-stabilizing no-op for this MLP architecture.")

    # ---- Step 5: same checks, but under real MXFP8 hardware quantization ----
    # This requires Blackwell (GB200/GB300); on Hopper (H100) MXFP8BlockScaling is
    # unavailable and this section is skipped with a clear message rather than a
    # crash, so this script still runs its bf16-only checks on cw for fast iteration.
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, MXFP8BlockScaling

        fp8_recipe = MXFP8BlockScaling(fp8_format=Format.HYBRID)
        HAVE_MXFP8 = True
    except Exception as exc:  # noqa: BLE001
        print(f"\n[step5] SKIPPED: MXFP8 unavailable on this hardware/TE version ({exc}).")
        HAVE_MXFP8 = False

    if HAVE_MXFP8:
        cfg_a2 = build_config(REAL_FFN)
        cfg_a2.fp8 = "hybrid"
        cfg_a2.fp8_recipe = "mxfp8"
        cfg_b2 = build_config(PAD_FFN)
        cfg_b2.fp8 = "hybrid"
        cfg_b2.fp8_recipe = "mxfp8"

        mlp_a2 = build_mlp(cfg_a2)
        mlp_b2 = build_mlp(cfg_b2)

        with torch.no_grad():
            mlp_b2.linear_fc1.weight.data.zero_()
            mlp_b2.linear_fc1.weight.data[:REAL_FFN, :].copy_(mlp_a2.linear_fc1.weight.data)
            mlp_b2.linear_fc1.bias.data.zero_()
            mlp_b2.linear_fc1.bias.data[:REAL_FFN].copy_(mlp_a2.linear_fc1.bias.data)
            mlp_b2.linear_fc2.weight.data.zero_()
            mlp_b2.linear_fc2.weight.data[:, :REAL_FFN].copy_(mlp_a2.linear_fc2.weight.data)
            if mlp_a2.linear_fc2.bias is not None:
                mlp_b2.linear_fc2.bias.data.copy_(mlp_a2.linear_fc2.bias.data)

        x2 = torch.randn(S, B, HIDDEN, device="cuda", dtype=torch.bfloat16)
        try:
            # step 5 expects TE's create_tensor to assert on the unpadded (4304)
            # tensor first -- that IS the bug we are working around, so mlp_a2
            # (the unpadded reference) is expected to raise here.
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out_a2, _ = mlp_a2(x2.clone())
            print("[step5] UNEXPECTED: unpadded (4304) MLP did not raise under MXFP8 -- "
                  "the original blocker may no longer apply, re-check assumptions.")
            unpadded_raised = False
        except (RuntimeError, AssertionError) as exc:
            print(f"[step5] unpadded (4304) MLP raises under MXFP8 as expected: {type(exc).__name__}")
            unpadded_raised = True

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out_b2, _ = mlp_b2(x2.clone())
        print(f"[step5] padded (4320) MLP runs under real MXFP8 autocast: output shape={tuple(out_b2.shape)}, "
              f"any_nan={torch.isnan(out_b2).any().item()}, any_inf={torch.isinf(out_b2).any().item()}")
        assert not torch.isnan(out_b2).any()
        assert not torch.isinf(out_b2).any()

        loss_b2 = out_b2.float().pow(2).sum()
        loss_b2.backward()
        g_w1_extra2 = mlp_b2.linear_fc1.weight.grad[REAL_FFN:, :]
        g_w2_extra2 = mlp_b2.linear_fc2.weight.grad[:, REAL_FFN:]
        print(f"[step5] under MXFP8, grad on padding: fc1.weight max_abs="
              f"{g_w1_extra2.abs().max().item()} fc2.weight max_abs={g_w2_extra2.abs().max().item()}")
        assert torch.equal(g_w1_extra2, torch.zeros_like(g_w1_extra2)), (
            "MXFP8 quantization broke the zero-padding invariant on fc1's backward"
        )
        assert torch.equal(g_w2_extra2, torch.zeros_like(g_w2_extra2)), (
            "MXFP8 quantization broke the zero-padding invariant on fc2's backward"
        )
        print(f"\nALL CHECKS PASSED (real MXFP8 path): padding stays an exact zero even under "
              f"hardware block quantization. unpadded_reference_raised_as_expected={unpadded_raised}")


if __name__ == "__main__":
    main()
