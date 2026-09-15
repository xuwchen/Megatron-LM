# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Download a pinned Nemotron Image v3 snapshot with a source-integrity manifest."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

_LOG = logging.getLogger(__name__)


def main() -> None:
    """Download the pinned HF repository with disk-space and file-size checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    dataset = 'nvidia/Nemotron-Image-Training-v3'
    revision = '7656391d4d4cb11ec3722b34f10d499435de0460'
    api = HfApi()
    info = api.dataset_info(dataset, revision=revision, files_metadata=True)
    files = []
    for item in info.siblings:
        lfs = item.lfs
        files.append(
            {
                'path': item.rfilename,
                'size': item.size,
                'sha256': lfs.sha256 if lfs else None,
                'git_blob_sha1': item.blob_id,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = args.output_dir.parent
    (state_dir / 'source-manifest.json').write_text(
        json.dumps({'dataset': dataset, 'revision': info.sha, 'files': files}, indent=2) + '\n'
    )
    missing_bytes = sum(
        f['size']
        for f in files
        if not (args.output_dir / f['path']).is_file()
        or (args.output_dir / f['path']).stat().st_size != f['size']
    )
    free = shutil.disk_usage(args.output_dir).free
    _LOG.info(
        json.dumps(
            {
                'phase': 'preflight',
                'files': len(files),
                'missing_bytes': missing_bytes,
                'free_bytes': free,
                'revision': info.sha,
            }
        )
    )
    if free < missing_bytes + 20 * 1024**3:
        raise RuntimeError('Insufficient free space for the snapshot and a 20 GiB reserve')
    # The small, self-contained Turing split makes an early real-data smoke test
    # available; SEC 1/2/4 follows the supplied example, then the rest of the repo.
    phases = [
        ('turing', ['turing/**']),
        ('sec124', ['long_document_sec_1/**', 'long_document_sec_2/**', 'long_document_sec_4/**']),
        ('full', None),
    ]
    for phase, patterns in phases:
        _LOG.info(json.dumps({'phase': phase, 'status': 'downloading', 'time': time.time()}))
        snapshot_download(
            repo_id=dataset,
            repo_type='dataset',
            revision=revision,
            local_dir=args.output_dir,
            allow_patterns=patterns,
            max_workers=args.workers,
        )
        selected = (
            files
            if patterns is None
            else [
                f
                for f in files
                if any(f['path'].startswith(p.removesuffix('**')) for p in patterns)
            ]
        )
        for f in selected:
            path = args.output_dir / f['path']
            if not path.is_file() or path.stat().st_size != f['size']:
                raise RuntimeError(f"Downloaded file size mismatch: {f['path']}")
        result = {
            'phase': phase,
            'status': 'downloaded',
            'revision': revision,
            'files': len(selected),
            'bytes': sum(f['size'] for f in selected),
            'time': time.time(),
        }
        (state_dir / f'{phase}-download-complete.json').write_text(
            json.dumps(result, indent=2) + '\n'
        )
        _LOG.info(json.dumps(result))


if __name__ == '__main__':
    main()
