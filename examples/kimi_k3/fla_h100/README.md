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

The accompanying GTP comparison also uses explicit FP64 gradient communication
and gradient-norm accumulation. Together, these controls passed the complete
100-update CW comparison: loss and gradient-norm TensorBoard scalars matched
at every update, with zero skipped
or NaN updates. This validates this explicit comparison mode, not default FP32
communication or bitwise identity of every final state. Parameter-norm statistics
retained a maximum relative difference of 1.493e-5. Fixing the gate launch
configuration alone has not been validated as a sufficient condition.
