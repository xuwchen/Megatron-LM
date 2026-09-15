# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Measure all Nemotron conversations at the training processor's image resolution.

Read image headers, run the real image processor once per unique resolution,
and expand its exact visual token count into the rendered ChatML token stream.
Validate this fast path against full pixel processing on a deterministic sample.
This is an eligibility census, not a model forward/backward benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
from PIL import Image

from examples.multimodal_dev.data.nemotron.processing import tokenize
from examples.multimodal_dev.data.nemotron.training_provider import (
    PROCESSOR,
    PROCESSOR_REVISION,
    REVISION,
)

_LOG = logging.getLogger(__name__)


def stats(values: list[int]) -> dict[str, float]:
    """Summarize a nonempty distribution with interpolated quantiles."""
    values = sorted(values)

    def quantile(p):
        x = (len(values) - 1) * p
        a = int(x)
        return values[a] + (values[min(a + 1, len(values) - 1)] - values[a]) * (x - a)

    return dict(
        min=values[0],
        mean=sum(values) / len(values),
        p50=quantile(0.5),
        p90=quantile(0.9),
        p99=quantile(0.99),
        max=values[-1],
    )


def expand_visual_tokens(ids: list[int], image_token: int, counts: list[int]) -> list[int]:
    """Expand one placeholder per image into its actual merged patch count."""
    if ids.count(image_token) != len(counts):
        raise ValueError('Rendered image placeholder count differs from source image count')
    result = []
    counts = iter(counts)
    for value in ids:
        result.extend([value] * next(counts) if value == image_token else [value])
    return result


def normalize_source_messages(
    messages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Match the converter's typed ChatML content and preserve image order."""
    normalized, references = [], []
    for message in messages:
        raw = message['content']
        parts = raw if isinstance(raw, list) else [raw]
        content = []
        for part in parts:
            if isinstance(part, str):
                content.append({'type': 'text', 'text': part})
            elif part['type'] == 'text':
                content.append({'type': 'text', 'text': part['text']})
            elif part['type'] == 'image':
                references.append(part['image'])
                content.append({'type': 'image'})
            else:
                raise ValueError(f'Unsupported Nemotron content type: {part}')
        normalized.append({'role': message['role'], 'content': content})
    return normalized, references


class MediaHeaders:
    """Index image bytes once and cache dimensions for the source length census."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.members = {}
        self.files = {}
        self.sizes = {}
        for path in sorted((directory / 'media').glob('*.tar')):
            with tarfile.open(path) as archive:
                for m in archive:
                    if m.isfile():
                        name = m.name.removeprefix('./')
                        if name in self.members:
                            raise ValueError(f'Duplicate media member: {name}')
                        self.members[name] = (path, m.offset_data, m.size)
        _LOG.info(f'Indexed {directory.name}: {len(self.members)} images')

    def stream(self, reference: str) -> tuple[BinaryIO, int]:
        """Seek to an image member and return its stream and byte length."""
        path, offset, size = self.members[reference]
        if path not in self.files:
            self.files[path] = path.open('rb')
        f = self.files[path]
        f.seek(offset)
        return f, size

    def size(self, reference: str) -> tuple[int, int]:
        """Read width and height from the image header, without decoding pixels."""
        if reference not in self.sizes:
            f, size = self.stream(reference)
            # PNG/JPEG headers fit in this window for these source archives;
            # Pillow raises on invalid/incomplete data instead of guessing.
            self.sizes[reference] = Image.open(io.BytesIO(f.read(min(size, 65536)))).size
        return self.sizes[reference]

    def image(self, reference: str) -> Image.Image:
        """Decode original image bytes for sampled processor parity checks."""
        f, size = self.stream(reference)
        return Image.open(io.BytesIO(f.read(size))).convert('RGB')

    def close(self) -> None:
        """Close cached archive file handles."""
        for f in self.files.values():
            f.close()


def main(default_subsets: list[str] | None = None) -> None:
    """Measure every record and validate sampled full-processor equivalence."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--subsets', nargs='+', default=default_subsets, required=default_subsets is None
    )
    parser.add_argument('--validate-per-subset', type=int, default=16)
    args = parser.parse_args()
    if args.validate_per_subset < 1:
        raise ValueError("At least one full-processing parity record per subset is required")
    torch.set_num_threads(4)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    marker = json.loads((args.data_root / 'full-download-complete.json').read_text())
    manifest = json.loads((args.data_root / 'source-manifest.json').read_text())
    if marker['revision'] != REVISION or manifest['revision'] != REVISION:
        raise ValueError('Source revision mismatch')
    expected = {x['path']: x for x in manifest['files']}
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        PROCESSOR, revision=PROCESSOR_REVISION, local_files_only=True
    )
    image_token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    grid_cache = {}
    started = time.monotonic()
    report = dict(
        revision=REVISION,
        processor=PROCESSOR,
        processor_revision=PROCESSOR_REVISION,
        min_pixels=4096,
        max_pixels=262144,
        subsets={},
        method=(
            'Full JSONL/image-header census; image grids from actual processor at each '
            'resolution; full pixel/token parity on hash-selected records'
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subset in args.subsets:
        directory = args.data_root / 'source' / subset
        source = directory / (subset + '.jsonl')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected[f'{subset}/{subset}.jsonl']['sha256']:
            raise ValueError(f'Source JSONL SHA-256 mismatch: {subset}')
        records = [json.loads(line) for line in source.open() if line.strip()]
        validation = set(
            sorted(
                range(len(records)),
                key=lambda i: hashlib.sha256(records[i]['id'].encode()).digest(),
            )[: args.validate_per_subset]
        )
        media = MediaHeaders(directory)
        rows = []
        checked = []
        try:
            for i, record in enumerate(records):
                messages, refs = normalize_source_messages(record['messages'])
                counts = []
                grids = []
                for ref in refs:
                    width, height = media.size(ref)
                    if (width, height) not in grid_cache:
                        encoded = processor.image_processor(
                            images=[Image.new('RGB', (width, height))],
                            return_tensors='pt',
                            min_pixels=4096,
                            max_pixels=262144,
                        )
                        grid_cache[width, height] = encoded['image_grid_thw'][0].tolist()
                    grid = grid_cache[width, height]
                    grids.append(grid)
                    counts.append(
                        int(torch.tensor(grid).prod()) // processor.image_processor.merge_size**2
                    )
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                raw_ids = processor.tokenizer(text, add_special_tokens=False)['input_ids']
                ids = expand_visual_tokens(raw_ids, image_token, counts)
                if i in validation:
                    sample = tokenize(processor, messages, [media.image(ref) for ref in refs])
                    if (
                        sample['input_ids'].tolist() != ids
                        or sample['image_grid_thw'].tolist() != grids
                    ):
                        raise ValueError(
                            f'Header census differs from full processing: {subset}/{record["id"]}'
                        )
                    checked.append(
                        dict(
                            id=record['id'],
                            tokens=len(ids),
                            images=len(refs),
                            targets=int(sample['loss_mask'].sum()),
                        )
                    )
                rows.append(
                    dict(
                        key=f'{subset}__{record["id"]}',
                        tokens=len(ids),
                        images=len(refs),
                        visual_tokens=sum(counts),
                        text_tokens=len(ids) - sum(counts),
                    )
                )
                if (i + 1) % 500 == 0:
                    _LOG.info(
                        f'{subset}: {i+1}/{len(records)} records; {len(grid_cache)} resolutions'
                    )
        finally:
            media.close()
        record_path = args.output_dir / f'{subset}-lengths.json'
        record_path.write_text(json.dumps(rows, separators=(',', ':')) + '\n')
        report['subsets'][subset] = dict(
            records=len(rows),
            source_sha256=digest,
            stats={
                k: stats([r[k] for r in rows])
                for k in ('tokens', 'images', 'visual_tokens', 'text_tokens')
            },
            eligible={
                str(n): sum(r['tokens'] <= n for r in rows) for n in (16384, 32768, 65536, 131072)
            },
            full_processing_parity=checked,
            lengths_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        )
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        _LOG.info(json.dumps({subset: report['subsets'][subset]}))
    _LOG.info('NEMOTRON_LENGTH_ASSESSMENT_PASSED')


if __name__ == '__main__':
    main()
