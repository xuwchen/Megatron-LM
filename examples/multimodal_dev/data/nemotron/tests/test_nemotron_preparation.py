# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integrity regressions for the pinned Nemotron converter adaptation."""
import hashlib
import json

import pytest

from examples.multimodal_dev.data.nemotron import prepare_nemotron_image_v3 as prep


def make_manifest(tmp_path):
    """Write two source payloads and their pinned hashes for corruption checks."""
    files = []
    for name, content in [
        ('sec/sec.jsonl', b'conversation'),
        ('sec/media/shard_000000.tar', b'archive'),
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(
            {'path': name, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        )
    manifest = {'dataset': prep.DATASET_ID, 'revision': prep.DATASET_REVISION, 'files': files}
    path = tmp_path / 'source-manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_api_manifest_verifies_payloads_and_detects_same_size_corruption(tmp_path):
    """Verify hashes detect corruption even when the file size still matches."""
    path, _ = make_manifest(tmp_path)
    pinned = prep._load_source_manifest(path, ('sec',))
    assert len(prep._verify_pinned_subset_files(tmp_path, ('sec',), pinned)) == 2
    (tmp_path / 'sec/media/shard_000000.tar').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='SHA-256'):
        prep._verify_pinned_subset_files(tmp_path, ('sec',), pinned)


@pytest.mark.parametrize('problem', ['revision', 'path', 'missing_media', 'missing_hash'])
def test_invalid_provenance_is_rejected_before_conversion(tmp_path, problem):
    """Reject invalid provenance before creating any conversion output."""
    path, manifest = make_manifest(tmp_path)
    if problem == 'revision':
        manifest['revision'] = 'another-revision'
    elif problem == 'path':
        manifest['files'][0]['path'] = '../outside'
    elif problem == 'missing_media':
        manifest['files'] = manifest['files'][:1]
    else:
        manifest['files'][0]['sha256'] = None
    path.write_text(json.dumps(manifest))
    output = tmp_path / 'output'
    with pytest.raises(ValueError):
        prep.prepare_nemotron_image_v3(tmp_path, output, subsets=('sec',), source_manifest=path)
    assert not output.exists()
