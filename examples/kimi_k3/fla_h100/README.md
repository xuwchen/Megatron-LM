# Fixed KDA gate backward configuration for numerical comparisons

FLA 0.5.2 autotunes the warp count of `kda_gate_bwd_kernel`. Changing the warp
count changes the FP32 reduction tree for `A_log` gradients, even when the
local gate gradients are identical. Megatron's deterministic mode alone does
not fix this external autotuner's choice.

Use FLA's supported cache configuration interface to keep this reduction
configuration identical on all ranks and across GTP layouts:

```bash
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR=/absolute/path/to/Megatron-LM/examples/kimi_k3/fla_h100
```

The JSON fixes only KDA gate backward to four warps and two stages. Other
kernels retain FLA's normal fallback behavior. This is a numerical comparison
configuration for the CW H100 Kimi-K3 slim proxy (32 heads, head dimension 64,
sequence length 4096), not a general performance recommendation. It does not
change the gate equation or disable its gradient computation. Keep FLA 0.5.2
pinned, since the cache schema and kernel names are dependency interfaces.

The current comparison uses ordinary FP32 gradient communication and norm
accumulation: leave `--grad-reduce-in-fp64` and `--grad-norm-in-fp64` disabled
(their defaults). A complete 100-update CW pair with this fixed gate setting
and both FP64 flags false passed its loss gates but failed 10 gradient-norm
gates. Maximum/mean loss absolute errors were 0.0198536 / 0.00208606, and the
maximum norm error relative to baseline was 9.7567%. Fixing this kernel alone
is therefore insufficient for the original per-update norm gate of
`0.001 + 0.05 * abs(baseline)`.

A historical pair using both FP64 controls together with this kernel setting
matched recorded loss and gradient-norm scalars across 100 updates. That
conditional diagnostic pass does not establish ordinary FP32-mode parity or
bitwise identity of every final state. These kernel settings have only been
examined on the CW H100 proxy and are not validated OCI performance settings.
