# Mock Variable-Length Datasets: Text and Multimodal

This note explains the text-only <code>MockVarlenDataset</code> added by
[NVIDIA/Megatron-LM#4832](https://github.com/NVIDIA/Megatron-LM/pull/4832),
compares it with ordinary mock data, and documents the Qwen3.5-VL multimodal
extension in this worktree.

The central difference is who owns final packing:

- <code>MockGPTDataset</code> emits fixed-length samples that the default
  collator stacks.
- #4832's <code>MockVarlenDataset</code> emits one minimally aligned text
  sample; the existing core DP x CP scheduler groups, reroutes, and packs it.
- <code>MockQwen35VLVarlenDataset</code> emits one unpadded image-text sample;
  the local multimodal packer concatenates token and vision fields along their
  different axes.

The current multimodal implementation exercises the existing local THD path.
It does not yet make #4832's global scheduler vision-aware.

## Terminology

| Symbol | Meaning |
|--------|---------|
| <code>S</code> | Configured maximum length of one sample |
| <code>L_i</code> | Valid multimodal token length of sample i |
| <code>P_i</code> | Raw vision-patch count of sample i |
| <code>V_i</code> | Merged vision-token/image-placeholder count of sample i |
| <code>D</code> | Flattened width of one raw vision patch |
| THD | Token-major packed attention layout |
| BSHD | Ordinary padded batch-major attention layout |

<code>cu_seqlens</code> counts logical, non-padding tokens.
<code>cu_seqlens_padded</code> records boundaries in physical THD storage
after alignment padding. They are equal only when no alignment padding was
inserted.

## 1. What PR #4832 Does

PR #4832 landed as commit <code>39f8d5524</code>, titled
<em>varlendataset for thd e2e and benchmark</em>. It does more than add one
mock class:

1. It adds <code>VarlenDataset</code> for real variable-length text and
   instruction data. Its low-level loader recognizes OpenAI messages,
   ShareGPT, Alpaca/Dolly, and plain pretraining-text schemas from Hugging Face
   or local parquet/JSON data.
2. It adds <code>MockVarlenDataset</code> for synthetic THD benchmarks.
3. It adds <code>--use-varlen-dataset</code>,
   <code>--varlen-mock-dataset-config-json</code>, and
   <code>--varlen-sbhd-validation</code>, along with argument validation and
   GPT dataset-config wiring.
4. It selects the varlen dataset from <code>pretrain_gpt.py</code>
   independently of the older <code>--sft</code> selector.
5. It selects an identity collator for variable-length and scheduler-driven
   paths, preserving a <code>list[dict]</code> instead of trying to stack
   unequal tensors.
6. It connects the one-sample-per-item contract to the existing packing
   schedulers and adds unit coverage.
7. It adds a real-data SBHD reference mode for comparing padded and packed
   execution with identical tokenization.

The scheduler classes already existed. #4832 integrates the new dataset with
them; it does not introduce the scheduler itself.

### 1.1 Text MockVarlenDataset contract

Each call to <code>MockVarlenDataset.__getitem__</code> returns one text
sample:

| Field | Dtype and shape | Purpose |
|-------|-----------------|---------|
| <code>tokens</code> | int64 <code>[P]</code> | Inputs after minimum alignment |
| <code>labels</code> | int64 <code>[P]</code> | Shifted next-token targets |
| <code>loss_mask</code> | float32 <code>[P]</code> | Ignored/padding-target mask |
| <code>position_ids</code> | int64 <code>[P]</code> | Per-sample text positions |
| <code>original_seq_len</code> | int32 <code>[1]</code> | Logical length |
| <code>padded_seq_len</code> | int32 <code>[1]</code> | Physical aligned length P |

The item is padded only to the minimum CP/SP alignment, not to
<code>S</code>. The scheduler later creates <code>cu_seqlens</code>,
<code>cu_seqlens_padded</code>, <code>max_seqlen</code>, and the final THD
buffer.

Static CP alignment is:

~~~text
(2 * CP if CP > 1 else 1) * SP_factor

SP_factor = TP when sequence parallelism is enabled, otherwise 1
~~~

Dynamic CP additionally includes DP in its alignment and scheduling rules.

### 1.2 Length-source modes and two implementation details

The text mock reuses <code>MockSFTLowLevelDataset</code>:

- <code>distribution</code>: bounded lognormal lengths;
- <code>file</code>: lengths loaded through pandas CSV;
- <code>verification</code>: token content comes from an
  <code>IndexedDataset</code>, while configured distribution parameters
  determine requested lengths.

The default distribution uses minimum <code>S/2</code>, maximum
<code>S</code>, mean <code>3S/4</code>, and sigma 1.1.

Two #4832 implementation details matter when reproducing exact examples:

1. File mode uses pandas' default header inference, so the first row of a
   truly headerless one-value-per-line file can be interpreted as a header.
2. A low-level requested length <code>L</code> produces
   <code>original_seq_len = L - 1</code> after EOD handling and next-token
   shifting.

Therefore, the output-level lengths 5, 3, and 9 in the example below require
file values 6, 4, and 10 in #4832. The new multimodal implementation
deliberately uses strict headerless CSV parsing and exact output lengths.

### 1.3 End-to-end ownership

~~~text
MockVarlenDataset
  |
  | one minimally aligned text sample
  | tokens / labels / loss_mask / position_ids
  | original_seq_len / padded_seq_len
  v
identity collate
  |
  | list[dict[str, Tensor]]
  v
existing core sequence-packing scheduler
  |-- gathers lengths across DP x CP
  |-- groups samples under the token budget
  |-- reroutes samples to assigned ranks
  |-- concatenates token-aligned fields
  +-- builds cu_seqlens / cu_seqlens_padded / max_seqlen
  v
packed language input [1, T] + PackedSeqParams(qkv_format="thd")
~~~

The dataset owns sample generation and minimum alignment. The scheduler owns
final global grouping and packing.

## 2. Ordinary Mock Data Versus Text MockVarlenDataset

### 2.1 Ordinary fixed-length mock

Let <code>S = 16</code> and read two samples:

~~~text
sample A: [A0 A1 A2 ... A15]   shape [16]
sample B: [B0 B1 B2 ... B15]   shape [16]

default collate

          sequence axis S=16
        <-------------------->
      +------------------------+
 B=2  | A0 A1 A2 ...       A15 |
      | B0 B1 B2 ...       B15 |
      +------------------------+

tokens.shape = [2, 16]
~~~

Every row occupies exactly S positions. No scheduler packing or cumulative
sequence lengths are needed to recover row boundaries.

### 2.2 Text-only variable-length example

Keep <code>S = 16</code>, use output-level lengths 5, 3, and 9, and set
alignment to 4:

~~~text
logical length     output after minimum alignment

A = 5              [A A A A A _ _ _]              original=5, padded=8
B = 3              [B B B _]                      original=3, padded=4
C = 9              [C C C C C C C C C _ _ _]    original=9, padded=12
~~~

Identity collate preserves unequal items:

~~~text
[
  {tokens: [8],  original_seq_len: [5], padded_seq_len: [8]},
  {tokens: [4],  original_seq_len: [3], padded_seq_len: [4]},
  {tokens: [12], original_seq_len: [9], padded_seq_len: [12]},
]
~~~

Suppose the scheduler groups A and B:

~~~text
physical THD storage

0                  8          12
|                  |           |
+------------------+-----------+
| A A A A A _ _ _  | B B B _   |
+------------------+-----------+

tokens.shape       = [1, 12]
cu_seqlens         = [0, 5, 8]    logical-token cumulative lengths
cu_seqlens_padded  = [0, 8, 12]   physical storage boundaries
max_seqlen         = 8
~~~

C is placed in another microbatch or grouped with later samples, depending on
the per-rank token budget.

### 2.3 Comparison with SFT paths

| Dataset | Per-item behavior | Boundary metadata | Final packing owner |
|---------|-------------------|-------------------|---------------------|
| <code>MockGPTDataset</code> | Fixed training window of S tokens | None | No packing |
| Real <code>SFTDataset</code> | May split/pack multiple conversations and emits dataset-side packed data | <code>cu_seqlens</code> | Dataset first; scheduler can unpack/repack |
| <code>MockSFTDataset</code> | One synthetic segment in THD mode, minimally aligned | One-segment <code>cu_seqlens</code> | Scheduler legacy path can unpack/repack |
| #4832 <code>MockVarlenDataset</code> | One unpacked, minimally aligned text sequence | Original and padded lengths | Core DP x CP scheduler |

The multiple-conversation behavior belongs to real
<code>SFTDataset</code>, not <code>MockSFTDataset</code>.

## 3. Why the Text Scheduler Is Not Multimodal-Ready

A Qwen3.5-VL sample has three independent ragged axes:

~~~text
language-token axis          raw-patch axis             image/video axis
-------------------          --------------             ----------------
input_ids      [L_i]         pixel_values [P_i, D]      image_grid_thw [N_i, 3]
labels         [L_i]
loss_mask      [L_i]

L_i, P_i, and N_i are generally different.
~~~

The current core scheduler contract is text-specific:

- required fields are <code>tokens</code>, <code>labels</code>,
  <code>loss_mask</code>, <code>position_ids</code>,
  <code>original_seq_len</code>, and <code>padded_seq_len</code>;
- the packing helper concatenates only token-aligned fields;
- pipeline-stage filtering removes metadata outside that known set;
- reroute/all-to-all sizing assumes non-length tensors follow the padded token
  length.

It cannot infer how many pixel or grid rows belong to a token sequence.
Treating every first dimension as a token dimension would break the
association between image placeholders and vision inputs.

Qwen3.5-VL also needs two ordering guarantees:

1. 3D MRoPE IDs must be computed from the final token order and image-grid
   order.
2. Vision embeddings must be produced and scattered into all matching image
   placeholders before language-token context-parallel splitting.

The multimodal provider therefore rejects the core
<code>--use-varlen-dataset</code> and
<code>--sequence-packing-scheduler</code> paths.

## 4. Multimodal MockVarlenDataset

### 4.1 Raw sample contract

<code>MockQwen35VLVarlenDataset</code> returns exactly five unpadded fields:

~~~text
{
  input_ids:       int64   [L_i],
  labels:          int64   [L_i],
  loss_mask:       float32 [L_i],
  pixel_values:    float32 [P_i, D],
  image_grid_thw:  int64   [N_i, 3],
}
~~~

It does not return position IDs or THD metadata. The local packer/model creates
them after the final packed order is known.

The mock generates one image per sample (<code>N_i = 1</code>). Without an
image-size config it keeps the fixed square <code>--image-size</code> behavior;
with bucket mode, image height and width vary deterministically per sample.
Videos and multiple images remain outside the current scope.

### 4.2 Geometry invariants

For grids <code>(T, H, W)</code> and merge factor <code>m</code>:

~~~text
P_i = sum(T * H * W)                  raw patches
V_i = sum(T * (H / m) * (W / m))      merged vision tokens
D   = 3 * temporal_patch_size * patch_size^2

count(input_ids == image_token_id) == V_i
~~~

Pixels have one row per raw patch. Input IDs have one placeholder per
post-merge vision token. The vision merger makes the final embedding count
equal the placeholder count.

For a still image, <code>T=1</code>. The processor duplicates the frame for
the temporal convolution, but <code>temporal_patch_size</code> is folded into
<code>D</code>, not into the grid's temporal-group count. Resolution buckets
describe processed <code>[height, width]</code> values and both dimensions
must be divisible by <code>patch_size * spatial_merge_size</code>.

### 4.3 Lengths and deterministic content

The multimodal mock accepts a bounded lognormal distribution or a strict
headerless CSV of integer output lengths. File lengths repeat cyclically.
Verification mode is omitted because text IndexedDataset lengths do not define
matching vision payloads.

Length, resolution, text-token, and pixel generation are deterministic
per index. The length is selected first; the resolution sampler cycles through
only those buckets whose merged-token count satisfies
<code>1 + V_i + 2 &lt;= L_i</code>. Thus exact CSV lengths are never clamped,
and every generated item retains at least two text tokens. Access order does
not change a sample, and long sequences do not produce out-of-vocabulary IDs.

Labels are next-token shifted. Terminal no-target positions and targets that
are multimodal special tokens become <code>-100</code>; the loss mask follows
these shifted targets. Thus the final IMG input may validly predict following
text.

### 4.4 Packing ownership

~~~text
MockQwen35VLVarlenDataset
  |
  | unpadded tokens + independent vision payloads
  v
identity collate (--use-vanilla-collate-fn)
  |
  | list[dict[str, Tensor]]
  v
multimodal pack_or_pad_batch
  |-- aligns token sequences for static CP/SP
  |-- concatenates token fields into [1, T]
  |-- concatenates pixels into [sum(P_i), D]
  |-- concatenates grids into [sum(N_i), 3]
  |-- builds cu_seqlens / cu_seqlens_padded / padding_mask
  |-- optionally aligns the final packed tail and appends a dummy sequence
  +-- broadcasts data and metadata across TP
  v
Qwen3.5-VL
  |-- computes packed MRoPE position_ids [3, 1, T]
  |-- encodes the complete vision payload
  |-- scatters merged embeddings into IMG positions
  +-- splits token-aligned tensors for static CP
  v
language decoder in THD format
~~~

Without <code>--use-packed-sequence</code>, the same provider enters the
padded BSHD branch instead.

## 5. Concrete Dynamic-Resolution Multimodal Example

Use two processed image-resolution buckets:

```text
patch_size          = 16
temporal_patch_size = 2
spatial_merge_size  = 2
resolutions         = [[32, 32], [32, 96]]

still-image T = 1
pixel width D = 3 * 2 * 16 * 16 = 1536
```

The bucket dimensions are multiples of
<code>patch_size * spatial_merge_size = 32</code>.

```text
sample A: 32 x 32                    sample B: 32 x 96
patch grid [1, 2, 2]                 patch grid [1, 2, 6]

+----+----+                           +----+----+----+----+----+----+
| A0 | A1 |                           | B0 | B1 | B2 | B3 | B4 | B5 |
+----+----+                           +----+----+----+----+----+----+
| A2 | A3 |                           | B6 | B7 | B8 | B9 |B10 |B11 |
+----+----+                           +----+----+----+----+----+----+
    |                                      |         |         |
    +-- one 2x2 merge                      +-- three 2x2 merges
        -> A.IMG0                              -> B.IMG0..2

A: P_A = 4,  V_A = 1
B: P_B = 12, V_B = 3
```

Use exact file-driven token lengths 7 and 11:

```text
sample A, L_A = 7
[t0 t1 VS IMG0 t2 t3 t4]

input_ids       [7]
pixel_values    [4, 1536]
image_grid_thw  [[1, 2, 2]]

sample B, L_B = 11
[u0 u1 u2 VS IMG0 IMG1 IMG2 u3 u4 u5 u6]

input_ids       [11]
pixel_values    [12, 1536]
image_grid_thw  [[1, 2, 6]]
```

<code>VS</code> is the vision-start token. Each <code>IMG</code> is one
post-merge placeholder.

### 5.1 CP=1, TP=1, sequence parallelism off

The alignment divisor is 1:

```text
0                       7                                      18
|                       |                                       |
+-----------------------+---------------------------------------+
|t0 t1 VS I0 t2 t3 t4   |u0 u1 u2 VS I0 I1 I2 u3 u4 u5 u6     |
+-----------------------+---------------------------------------+
          A: 7                           B: 11

input_ids.shape       = [1, 18]
labels.shape          = [1, 18]
loss_mask.shape       = [1, 18]
padding_mask.shape    = [1, 18], all false
cu_seqlens            = [0, 7, 18]
cu_seqlens_padded     = [0, 7, 18]
max_seqlen            = 11
total_tokens          = 18

pixel_values.shape    = [4 + 12, 1536] = [16, 1536]
image_grid_thw        = [[1, 2, 2], [1, 2, 6]]
image_grid_thw.shape  = [2, 3]
IMG positions         = 1 + 3 = 4
```

Token lengths, raw-patch lengths, and grid rows are three independent ragged
axes.

### 5.2 CP=2, sequence parallelism off

The static CP divisor is <code>2 * CP = 4</code>:

```text
A: 7  -> 8 physical positions
B: 11 -> 12 physical positions

A = [t0 t1 VS I0 t2 t3 t4 PAD]
B = [u0 u1 u2 VS I0 I1 I2 u3 u4 u5 u6 PAD]

input_ids.shape       = [1, 20]
cu_seqlens            = [0, 7, 18]
cu_seqlens_padded     = [0, 8, 20]
padding_mask          = true at physical positions 7 and 19
max_seqlen            = 12
total_tokens          = 20
```

The vision shapes remain <code>[16, 1536]</code> and <code>[2, 3]</code>:
language padding never creates patches, grids, or IMG placeholders.

With the existing zigzag rule, each aligned sequence is divided into
<code>2 * CP = 4</code> chunks:

| sample | chunk | content | owner |
|---|---|---|---:|
| A | A.c0 | <code>t0 t1</code> | CP rank 0 |
| A | A.c1 | <code>VS IMG0</code> | CP rank 1 |
| A | A.c2 | <code>t2 t3</code> | CP rank 1 |
| A | A.c3 | <code>t4 PAD</code> | CP rank 0 |
| B | B.c0 | <code>u0 u1 u2</code> | CP rank 0 |
| B | B.c1 | <code>VS IMG0 IMG1</code> | CP rank 1 |
| B | B.c2 | <code>IMG2 u3 u4</code> | CP rank 1 |
| B | B.c3 | <code>u5 u6 PAD</code> | CP rank 0 |

The 7/11 integration test directly covers CP=1. The CP=2 alignment metadata
is also covered directly; model/Transformer Engine tests own the later zigzag
partition.

### 5.3 Final packed-tail alignment at 128

With <code>--pad-packed-seq-alignment 128</code> and
<code>--pad-packed-seq-by-appending-dummy-seq</code>, the local multimodal
packer applies a second, final-tail alignment after the per-sample CP/SP
alignment. For CP=1:

```text
physical token rows       18 -> 128
cu_seqlens                [0, 7, 18, 128]
cu_seqlens_padded         [0, 7, 18, 128]
padding_mask=true         [18, 128)
```

For CP=2, the real samples first become 8 and 12 global physical positions.
The alignment value is CP-local: the existing local length is 20/2=10, so each
rank pads to 128 and the global target is 256:

```text
CP-local token rows       10 -> 128
global physical rows      20 -> 256
cu_seqlens                [0, 7, 18, 254]
cu_seqlens_padded         [0, 8, 20, 256]
padding_mask=true         position 7, position 19, and [20, 256)
```

The appended boundary is a dummy language sequence with physical length 236.
Its logical endpoint is 18+236=254, while its padded endpoint is 256; this
keeps MRoPE's valid length equal to the physical dummy segment even though the
real samples already contain two alignment slots. Its token IDs and loss mask
are neutral, while pixels remain <code>[16,1536]</code> and grids remain
<code>[2,3]</code>. Numeric alignment 128 is compatible with
<code>2 * CP</code> for the CP sizes used by these recipes.

## 6. Language THD and Vision THD Are Different

```text
language THD
  boundaries: cu_seqlens / cu_seqlens_padded
  elements:   text tokens + image placeholders + optional PAD

vision-encoder THD
  boundaries: derived from image_grid_thw
  elements:   raw vision patches
```

For the A/B example:

| axis | boundaries | maximum subsequence |
|---|---|---:|
| Language, CP=1 | <code>[0, 7, 18]</code> | 11 tokens |
| Language, CP=2 physical | <code>[0, 8, 20]</code> | 12 positions |
| Vision | <code>[0, 4, 16]</code> | 12 raw patches |

The still-image grids have <code>T=1</code>, so each image contributes one
vision subsequence:

| image | grid | raw-patch interval | merge outputs |
|---|---|---|---|
| A | <code>[1,2,2]</code> | <code>[0,4)</code> | <code>A.IMG0</code> |
| B | <code>[1,2,6]</code> | <code>[4,16)</code> | <code>B.IMG0..2</code> |

The two THD layouts are connected only by the invariant that the ordered
merged vision embeddings equal the ordered IMG placeholders. Their cumulative
length arrays are not interchangeable.

## 7. What Changed in This Worktree

### examples/multimodal_dev/data/mock_varlen.py

- Adds the deterministic length sampler and strict headerless CSV parser.
- Adds deterministic processed-resolution buckets with fixed
  <code>--image-size</code> fallback.
- Filters buckets by each exact <code>L_i</code> so every sample retains at
  least two text tokens.
- Uses still-image <code>T=1</code> while keeping
  <code>temporal_patch_size</code> in the pixel width <code>D</code>.
- Adds <code>MockQwen35VLVarlenDataset</code> with the five-field contract.
- Derives and validates patch, placeholder, and pixel shapes.
- Builds shifted labels and target-aligned loss masks.
- Adds train/validation/test providers with independent seeds.
- Fails fast for invalid geometry, core scheduler flags, missing identity
  collate, unequal sequence-length arguments, or packed HybridEP without
  group-wide variable-token padding.

### examples/multimodal_dev/arguments.py

- Lists <code>mock_varlen</code> in dataset-provider help.
- Adds <code>--mock-image-size-config-json</code> for inline or file-based
  resolution-bucket configuration.

### examples/multimodal_dev/tests

- CPU tests cover fixed fallback, dynamic variation, determinism, exact-length
  feasibility, geometry invariants, and invalid configs.
- The THD integration test uses the 32x32 and 32x96 samples above and verifies
  combined pixels <code>[16,1536]</code>, grids <code>[2,3]</code>, and four
  IMG placeholders.
- Additional THD tests cover 128-token final-tail alignment and an appended
  dummy sequence for both CP=1 and CP=2 metadata.

### examples/multimodal_dev/forward_step.py

<code>pack_or_pad_batch</code> still owns loader-microbatch multimodal packing.
It now includes a pre-CP adapter for the standard packed-padding CLI flags.
The adapter resolves numeric alignment in CP-local coordinates, pads the
global buffer, and gives the dummy sequence distinct logical/padded endpoints
when real samples already contain alignment slots. It does not replace this
local path with the core #4832 scheduler.

## 8. Launch Configuration and Boundaries

Minimal packed-THD selection with dynamic processed resolutions:

```bash
--dataset-provider mock_varlen \
--seq-length 32768 \
--total-seq-length 32768 \
--use-vanilla-collate-fn \
--use-packed-sequence \
--linear-cp-mode chunkwise \
--max-seqlen-per-dp-cp-rank 32768 \
--pad-packed-seq-alignment 128 \
--pad-packed-seq-by-appending-dummy-seq \
--moe-hybridep-pad-variable-tokens \
--varlen-mock-dataset-config-json \
  '{"mode":"distribution","type":"lognormal","min_seq_len":1024,"max_seq_len":32768,"mean_seq_len":8192,"lognormal_sigma":1.1}' \
--mock-image-size-config-json \
  '{"mode":"buckets","resolutions":[[224,224],[224,448],[448,224],[448,448]]}'
```

The image-size config accepts an inline JSON object or a path to a JSON file.
Bucket dimensions are post-processor pixel sizes and must be aligned to
<code>patch_size * spatial_merge_size</code>. Omitting it preserves the fixed
square <code>--image-size</code> path.

The HybridEP flag is required only with
<code>--moe-token-dispatcher-type flex</code> and
<code>--moe-flex-dispatcher-backend hybridep</code>. Local THD packing can
produce different token counts across the HybridEP group; this option pads to
the group maximum before dispatch and trims after combine. The provider
rejects that HybridEP combination when the flag is absent.

Do not combine this provider with:

```text
--use-varlen-dataset
--sequence-packing-scheduler ...
--dynamic-context-parallel
```

The first two select the text-only core contract. Dynamic CP depends on the
global scheduler, so this multimodal path supports static CP only. An existing
<code>--sft</code> setting can remain for reporting semantics.

Current scope:

- Packing is local to one loader microbatch, not global bin-packing across
  DP x CP ranks.
- <code>seq_length</code> limits one item, not the combined packed
  <code>T</code>.
- One still image is generated per sample; its processed H/W can vary by
  bucket. Multi-image samples and video remain out of scope.
- The 7/11 integration test directly covers CP=1, CP=2 static alignment, and
  optional 128-token final-tail padding. Model/TE tests separately own the
  actual CP zigzag partition.

## 9. Summary

~~~text
ordinary mock
  fixed [S] item -> stack [B, S]

#4832 text mock varlen
  aligned [P_i] item -> global core schedule/reroute -> packed [1, T]

current multimodal mock varlen
  raw tokens [L_i] + pixels [P_i, D] + grids [N_i, 3]
  -> local three-axis multimodal packing -> packed language THD
~~~

The implementation is a multimodal variable-length source for the existing
Qwen3.5-VL local THD path, not a vision-aware replacement for the core #4832
scheduler.
