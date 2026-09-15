# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepare any self-contained Nemotron subset through one admission pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_LOG = logging.getLogger(__name__)


def main(default_subset: str | None = None, default_prepared_name: str | None = None) -> None:
    """Run source verification, conversion, full census, and selection validation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--subset', default=default_subset, required=default_subset is None)
    p.add_argument('--prepared-dir', type=Path)
    p.add_argument('--assessment-dir', type=Path)
    p.add_argument('--token-budget', type=int, default=32768)
    p.add_argument('--validate-samples', type=int, default=16)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--inside-venv', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.subset):
        raise ValueError('Subset must be a single safe directory name')
    if min(args.token_budget, args.validate_samples, args.workers) < 1:
        raise ValueError('Token budget, validation count and workers must be positive')
    if not args.inside_venv:
        venv = Path(tempfile.mkdtemp(prefix='nemotron-unified-prep-')) / 'venv'
        subprocess.run(
            [sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], check=True
        )
        python = str(venv / 'bin/python')
        subprocess.run(
            [
                python,
                '-m',
                'pip',
                'install',
                '--disable-pip-version-check',
                'webdataset==1.0.2',
                'megatron-energon==7.4.0',
                # Match the inherited datasets 4.8.4 fsspec upper bound.
                'fsspec==2026.2.0',
            ],
            check=True,
        )
        # Replay this same entry point so legacy defaults remain compatible.
        subprocess.run([python, sys.argv[0], *sys.argv[1:], '--inside-venv'], check=True)
        return
    project = Path(__file__).resolve().parents[4]
    scripts = Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    from examples.multimodal_dev.data.nemotron.prepare_nemotron_image_v3 import (
        _load_source_manifest,
        _verify_pinned_subset_files,
    )
    from examples.multimodal_dev.data.nemotron.training_provider import (
        REVISION,
        build_index,
        verified_manifest,
    )

    marker = json.loads((args.data_root / 'full-download-complete.json').read_text())
    if marker['revision'] != REVISION:
        raise ValueError('Download completion revision mismatch')
    source_manifest = args.data_root / 'source-manifest.json'
    prepared = args.prepared_dir or args.data_root / (
        default_prepared_name or f'energon-{args.subset}'
    )
    assessment = args.assessment_dir or prepared / 'mdp-assessment' / args.subset
    source_files = _load_source_manifest(source_manifest, (args.subset,))
    verified = _verify_pinned_subset_files(args.data_root / 'source', (args.subset,), source_files)
    if not (prepared / 'manifest.json').exists():
        subprocess.run(
            [
                sys.executable,
                str(scripts / 'prepare_nemotron_image_v3.py'),
                '--source-dir',
                str(args.data_root / 'source'),
                '--output-dir',
                str(prepared),
                '--source-manifest',
                str(source_manifest),
                '--subsets',
                args.subset,
                '--validation-fraction',
                '0.05',
                '--max-samples-per-tar',
                '1000',
                '--num-workers',
                str(args.workers),
            ],
            check=True,
        )
    manifest = verified_manifest(prepared)
    if args.subset not in manifest['subsets']:
        raise ValueError('Prepared directory does not contain the requested subset')
    recorded = {r['path']: r for r in manifest['verified_source_files']}
    if any(recorded.get(r['path']) != r for r in verified):
        raise ValueError('Prepared manifest differs from verified source payloads')
    build_index(prepared)
    _LOG.info(f'FULL_CENSUS_REQUIRED: subset={args.subset} budget={args.token_budget}')
    subprocess.run(
        [
            sys.executable,
            str(scripts / 'assess_records.py'),
            '--data-root',
            str(args.data_root),
            '--output-dir',
            str(assessment),
            '--subsets',
            args.subset,
            '--validate-per-subset',
            str(args.validate_samples),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / 'prepare_selection.py'),
            '--prepared-dir',
            str(prepared),
            '--assessment-dir',
            str(assessment),
            '--subset',
            args.subset,
            '--token-budget',
            str(args.token_budget),
            '--validate-samples',
            str(args.validate_samples),
        ],
        check=True,
    )
    selection_path = prepared / f'mdp-selection-{args.subset}-{args.token_budget}.json'
    selection = json.loads(selection_path.read_text())
    receipt = {
        'kind': 'nemotron-unified-preparation',
        'revision': REVISION,
        'subset': args.subset,
        'token_budget': args.token_budget,
        'selection_path': str(selection_path),
        'assessment_dir': str(assessment),
        'counts': selection['counts'],
        'full_census_required': True,
        'source_integrity_verified': True,
        'provider_validation_passed': True,
    }
    selection_path.with_suffix('.preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    _LOG.info('NEMOTRON_UNIFIED_PREPARATION_PASSED: ' + json.dumps(receipt))


if __name__ == '__main__':
    main()
