# BF16 HF arithmetic parity

`hf_parity_gtp4.yaml` enables the explicit operation order used by the pinned
Moonshot Kimi-K3 reference. Combine it with the 29-layer slim model definition
from the correctness kit. The optimized training defaults remain unchanged.

| Option | Effect |
|---|---|
| `rmsnorm_impl: torch` | FP32 RMS normalization, input-dtype cast, then scale; preserves norm/linear checkpoint keys and GTP linear sharding |
| `attn_res_impl: torch` | Normalize depth keys before scoring and use batched weighted aggregation |
| `native_situ_glu: true` | Eager FP32 SiTU operations, then input-dtype cast |
| `moe_router_use_torch_linear: true` | Cast both router operands to FP32 before the linear operation |
| `moe_apply_probs_on_output: true` | Weight complete expert FC2 outputs and combine in FP32, then cast before latent-up normalization |

Use FlashAttention 2 on both implementations. In the recorded Transformer Engine
runtime, set `NVTE_FLASH_ATTN_V2=1`, `NVTE_FLASH_ATTN_V3=0`, and
`NVTE_FLASH_ATTN_V4=0` in addition to `attention_backend: flash`.
Do not fix the FLA gate configuration or enable FP64 gradients/norms.

The independent reference is `moonshotai/Kimi-K3` revision
`a590ce090cb049c93a33dfe8c208ec652aa20503`, file `modeling_kimi_linear.py`.
The associated checker is `yuzhongw/mcore-kit-for-kimi-k3` revision
`5f7d6f71883d6b19989adf403f3e802eb5b3fcaf`,
`correctness_check/kimi_k3_proxy_9layer_one_step.py`.

Native order adds kernels and FP32 saved activations. Post-FC2 probability
weighting uses FP32 expert combine communication (twice the BF16 payload).
This recipe supports unquantized, bias-free experts with the all-to-all dispatcher
and shared-expert overlap disabled; it does not apply to the MXFP8 performance
recipe. These options require their own performance measurements.

Natural routing and replayed HF top-k indices are separate checks. Replay
retains MCore's differentiable probabilities. The pinned HF inference MoE is
no-grad, so complete-model forward parity does not establish end-to-end training
gradient or convergence parity. Unit tests separately compare input and parameter
gradients against independent differentiable formulas, including recomputation.
