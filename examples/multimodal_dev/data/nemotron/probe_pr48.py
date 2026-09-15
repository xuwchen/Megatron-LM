# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise PR 48 packing and CUDA collation with real Nemotron image samples.

This is a data-path probe, not a model forward/backward or training benchmark.
Only read datasets produced by our converter: jpgs contains its own trusted
pickle serialization of image byte strings.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import pickle
import sys
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from examples.multimodal_dev.data.nemotron.processing import tokenize

_LOG = logging.getLogger(__name__)


def prepared_records(
    root: Path, limit: int, samples_per_subset: int | None = None
) -> Iterator[tuple[str, list[dict[str, Any]], list[Image.Image]]]:
    """Read converter-owned records, optionally covering each requested subset."""
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest.get('source_integrity_verified'):
        raise ValueError('The probe requires verified prepared data')
    selected = {subset: 0 for subset in manifest['subsets']}
    if samples_per_subset is not None:
        if samples_per_subset <= 0:
            raise ValueError('samples_per_subset must be positive')
        limit = len(selected) * samples_per_subset
    count = 0
    for path in sorted(root.glob('train-shard-*.tar')):
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.name.endswith('.json'):
                    continue
                key = member.name.removesuffix('.json')
                subset = key.partition('__')[0]
                if subset not in selected:
                    raise ValueError(f'Unexpected subset in prepared sample key: {key}')
                if samples_per_subset is not None and selected[subset] >= samples_per_subset:
                    continue
                messages = json.load(archive.extractfile(member))
                images = pickle.loads(archive.extractfile(key + '.jpgs').read())
                if not isinstance(images, list) or not all(isinstance(x, bytes) for x in images):
                    raise ValueError('Unexpected converted image payload')
                yield key, messages, [Image.open(io.BytesIO(x)).convert('RGB') for x in images]
                count += 1
                selected[subset] += 1
                if count == limit:
                    return
    if samples_per_subset is not None:
        raise ValueError(f'Could not fill requested subset coverage: {selected}')


def main() -> None:
    """Check real-data greedy packing and CUDA collation without model training."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=int, default=600)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument(
        '--samples-per-subset',
        type=int,
        help='Select this many records from every prepared subset, replacing --samples',
    )
    parser.add_argument('--token-budget', type=int, default=32768)
    parser.add_argument('--max-sequences', type=int, default=16)
    args = parser.parse_args()
    from transformers import AutoProcessor

    deadline = time.monotonic() + args.wait_seconds
    while not (args.data_dir / 'manifest.json').is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('Verified preparation did not finish in time for the GPU probe')
        _LOG.info('Waiting for verified prepared data')
        time.sleep(15)
    model = 'Qwen/Qwen3.5-35B-A3B'
    revision = '59d61f3ce65a6d9863b86d2e96597125219dc754'
    processor = AutoProcessor.from_pretrained(model, revision=revision)
    records = list(prepared_records(args.data_dir, args.samples, args.samples_per_subset))
    samples = []
    for i, (key, messages, images) in enumerate(records):
        sample = tokenize(processor, messages, images)
        sample['sample_key'] = key
        samples.append(sample)
        _LOG.info(
            json.dumps(
                {
                    'sample': i,
                    'tokens': len(sample['input_ids']),
                    'images': len(images),
                    'pixel_rows': len(sample['pixel_values']),
                }
            )
        )
    if not samples:
        raise ValueError('No real samples loaded')
    subset_counts = {}
    for sample in samples:
        subset = sample['sample_key'].partition('__')[0]
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    _LOG.info(json.dumps({'selected_subsets': subset_counts}))
    from examples.multimodal_dev import forward_step as fs
    from megatron.core import parallel_state as mpu
    from megatron.core.mdp.packing import GreedySampleStream, decoder_sample_length
    from megatron.training.global_vars import set_args

    image_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    set_args(
        SimpleNamespace(
            sequence_parallel=False,
            mdp_enable=True,
            thd_static_packing=True,
            max_seqlen_per_dp_cp_rank=args.token_budget,
            thd_max_packed_sequences=args.max_sequences,
            thd_tail_padding_policy='append_dummy_seq',
            image_token_id=image_token_id,
            vision_spatial_merge_size=processor.image_processor.merge_size,
        )
    )
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    torch.distributed.init_process_group('nccl')
    mpu.initialize_model_parallel()
    try:
        stream = GreedySampleStream(
            iter([samples[i : i + 4] for i in range(0, len(samples), 4)]),
            token_budget=args.token_budget,
            max_num_seqs=args.max_sequences - 1,
            align=1,
            length_of=decoder_sample_length,
        )
        bins = []
        seen = []
        for batch in stream:
            lengths = [len(s['input_ids']) for s in batch]
            real = sum(lengths)
            packed = fs.pack_or_pad_batch(
                batch, use_packed_sequence=True, with_vision_sidecar=True, device='cuda'
            )
            for field in ('input_ids', 'labels', 'loss_mask', 'padding_mask'):
                assert tuple(packed[field].shape) == (1, args.token_budget), field
            for field in ('input_ids', 'labels', 'loss_mask'):
                expected = torch.cat([s[field] for s in batch])
                assert torch.equal(packed[field][0, :real].cpu(), expected), field
            for field in ('pixel_values', 'image_grid_thw'):
                assert torch.equal(packed[field].cpu(), torch.cat([s[field] for s in batch])), field
            assert bool((packed['labels'][0, real:] == -100).all())
            assert bool((packed['loss_mask'][0, real:] == 0).all())
            assert bool(packed['padding_mask'][0, real:].all())
            metadata = packed['packed_seq_params']
            for field in (
                'cu_seqlens_q',
                'cu_seqlens_kv',
                'cu_seqlens_q_padded',
                'cu_seqlens_kv_padded',
            ):
                value = getattr(metadata, field)
                assert value.numel() == args.max_sequences + 1, field
                assert int(value[-1]) == args.token_budget, field
            expected_cu = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32
            )
            assert torch.equal(metadata.cu_seqlens_q[: len(lengths) + 1].cpu(), expected_cu)
            assert metadata.max_seqlen_q == args.token_budget
            assert metadata.pad_between_seqs is False
            stream.commit(len(batch))
            seen.extend(s['sample_key'] for s in batch)
            row = {
                'samples': len(batch),
                'lengths': lengths,
                'real_tokens': real,
                'padded_tokens': args.token_budget - real,
                'shape': list(packed['input_ids'].shape),
                'cu_seqlens_shape': list(metadata.cu_seqlens_q.shape),
                'images': sum(len(s['image_grid_thw']) for s in batch),
            }
            bins.append(row)
            _LOG.info(json.dumps({'bin': len(bins) - 1, **row}))
        assert seen == [
            s['sample_key'] for s in samples
        ], 'Samples were reordered, duplicated, or lost'
        assert stream.consumed_samples == len(samples)
        assert any(b['samples'] > 1 for b in bins), 'No multi-sample packing observed'
        assert any(b['padded_tokens'] > 0 for b in bins), 'No static tail observed'
        report = {
            'status': 'passed',
            'scope': (
                'real-image tokenization, greedy packing and CUDA collation; ' 'no model training'
            ),
            'pr_head': '682d23e484805ccb9833bfbead795c96c4f239fb',
            'processor': model,
            'processor_revision': revision,
            'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(),
            'data_dir': str(args.data_dir),
            'image_min_pixels': 4096,
            'image_max_pixels': 262144,
            'samples': len(samples),
            'sample_counts_by_subset': subset_counts,
            'bins': bins,
            'consumed_samples': stream.consumed_samples,
            'padding_fraction': sum(b['padded_tokens'] for b in bins)
            / (len(bins) * args.token_budget),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        _LOG.info(json.dumps(report))
    finally:
        mpu.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
