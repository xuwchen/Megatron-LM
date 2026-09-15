# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Build a complete-record Nemotron selection over existing verified MDP tar indexes."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from examples.multimodal_dev.data.nemotron.training_provider import (
    PROCESSOR,
    PROCESSOR_REVISION,
    NemotronDataset,
    build_index,
    select_records,
)

_LOG = logging.getLogger(__name__)


def make_selection(
    index: dict[str, Any],
    summary: dict[str, Any],
    lengths: list[dict[str, Any]],
    *,
    subset: str,
    token_budget: int,
) -> dict[str, Any]:
    """Admit whole records from a complete census and preserve original split order."""
    if (
        summary['processor'],
        summary['processor_revision'],
        summary['min_pixels'],
        summary['max_pixels'],
    ) != (PROCESSOR, PROCESSOR_REVISION, 4096, 262144):
        raise ValueError('Length census uses a different processor configuration')
    source = next(
        x
        for x in index['manifest']['verified_source_files']
        if x['path'] == f'{subset}/{subset}.jsonl'
    )
    if summary['subsets'][subset]['source_sha256'] != source['sha256']:
        raise ValueError('Length census and prepared source SHA-256 differ')
    by_key = {r['key']: r for r in lengths}
    expected_keys = {
        r['key']
        for records in index['splits'].values()
        for r in records
        if r['key'].partition('__')[0] == subset
    }
    if len(by_key) != len(lengths) or set(by_key) != expected_keys:
        raise ValueError('Length census must cover each selected-subset record exactly once')
    selected, counts = {}, {}
    for split, records in index['splits'].items():
        candidates = [r for r in records if r['key'] in by_key]
        selected[split] = [
            r['key'] for r in candidates if 0 < by_key[r['key']]['tokens'] <= token_budget
        ]
        counts[split] = dict(
            source_records=len(candidates),
            selected_records=len(selected[split]),
            excluded_over_budget=sum(by_key[r['key']]['tokens'] > token_budget for r in candidates),
        )
        if not selected[split] or any(by_key[r['key']]['tokens'] <= 0 for r in candidates):
            raise ValueError('Empty split or invalid record length')
    selection = dict(
        kind='nemotron-mdp-complete-record-selection',
        format_version=1,
        manifest=index['manifest'],
        subset=subset,
        token_budget=token_budget,
        processor=PROCESSOR,
        processor_revision=PROCESSOR_REVISION,
        min_pixels=4096,
        max_pixels=262144,
        counts=counts,
        splits=selected,
    )
    select_records(index, selection, token_budget)
    return selection


def publish_selection(path: str | Path, selection: dict[str, Any]) -> None:
    """Keep identical admission files stable and publish new ones atomically."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise ValueError(
                'Existing selection differs; use a separate prepared directory '
                'for another provenance'
            )
        return
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(selection, separators=(',', ':')) + '\n')
    temporary.replace(path)


def main(default_subset: str | None = None) -> None:
    """Validate actual Dataset outputs before publishing a budget-bound selection."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--assessment-dir', type=Path, required=True)
    p.add_argument('--subset', default=default_subset, required=default_subset is None)
    p.add_argument('--token-budget', type=int, default=32768)
    p.add_argument('--validate-samples', type=int, default=16)
    args = p.parse_args()
    if args.validate_samples < 1 or args.token_budget < 1:
        raise ValueError("Positive token budget and provider validation sample count are required")
    index = build_index(args.prepared_dir)
    summary = json.loads((args.assessment_dir / 'summary.json').read_text())
    lengths_path = args.assessment_dir / (args.subset + '-lengths.json')
    digest = hashlib.sha256(lengths_path.read_bytes()).hexdigest()
    if summary['subsets'][args.subset]['lengths_sha256'] != digest:
        raise ValueError('Length census SHA-256 mismatch')
    lengths = json.loads(lengths_path.read_text())
    selection = make_selection(
        index, summary, lengths, subset=args.subset, token_budget=args.token_budget
    )
    selection['lengths_sha256'] = digest
    selected = select_records(index, selection, args.token_budget)
    checked = {}
    for split, records in selected.items():
        ds = NemotronDataset(
            args.prepared_dir,
            records,
            target_length=len(records),
            max_length=args.token_budget,
            vocab_size=248320,
            image_token_id=248056,
        )
        # Full processor validation, including near-budget records as well as
        # hash-selected records, against the source census used for eligibility.
        expected = {r['key']: r['tokens'] for r in lengths}
        choices = sorted(
            range(len(records)), key=lambda i: hashlib.sha256(records[i]['key'].encode()).digest()
        )[: args.validate_samples]
        choices += sorted(
            range(len(records)), key=lambda i: expected[records[i]['key']], reverse=True
        )[:2]
        checked[split] = []
        for i in dict.fromkeys(choices):
            sample = ds[i]
            if len(sample['input_ids']) != expected[records[i]['key']]:
                raise ValueError('Prepared training tokenization differs from source length census')
            checked[split].append(
                dict(
                    key=records[i]['key'],
                    tokens=len(sample['input_ids']),
                    images=len(sample['image_grid_thw']),
                    targets=int(sample['loss_mask'].sum()),
                )
            )
        _LOG.info(json.dumps({split: selection['counts'][split], 'validated': checked[split]}))
    path = args.prepared_dir / f'mdp-selection-{args.subset}-{args.token_budget}.json'
    publish_selection(path, selection)
    path.with_suffix('.validation.json').write_text(json.dumps(checked, indent=2) + '\n')
    _LOG.info(f'NEMOTRON_TRAINING_SELECTION_PASSED: {path}')


if __name__ == '__main__':
    main()
