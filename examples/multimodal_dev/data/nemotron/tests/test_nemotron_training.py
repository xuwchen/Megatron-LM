# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regressions for the real-data contract handed to PR 48."""
import io
import json
import pickle
import tarfile
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.multimodal_dev.data.nemotron import training_provider as provider
from examples.multimodal_dev.data.nemotron.processing import tokenize


class Processor:
    """Expose fixed ChatML tokens with known target spans and one image grid."""

    image_token = '<image>'
    image_processor = SimpleNamespace(merge_size=2)
    # Two assistant turns, a masked user image token, and a special reasoning
    # delimiter inside the first answer. Numbers 20/21/22 are answer targets.
    ids = [1, 11, 9, 30, 2, 1, 10, 3, 20, 21, 2, 1, 11, 31, 2, 1, 10, 22, 2]

    def __init__(self):
        self.tokenizer = self
        self.all_special_ids = [1, 2, 3, 9]

    def apply_chat_template(self, messages, **kwargs):
        """Return rendered text independently of raw answer strings."""
        return 'rendered conversation'

    def convert_tokens_to_ids(self, value):
        """Resolve the role boundaries and image placeholder used by the fixture."""
        return {'<|im_start|>': 1, '<|im_end|>': 2, '<image>': 9}[value]

    def encode(self, text, **kwargs):
        """Supply the tokenized assistant role header."""
        assert text == 'assistant\n'
        return [10]

    def __call__(self, **kwargs):
        return {
            'input_ids': torch.tensor([self.ids]),
            'pixel_values': torch.zeros(4, 12),
            'image_grid_thw': torch.tensor([[1, 2, 2]]),
        }


MESSAGES = [
    {'role': role, 'content': [{'type': 'text', 'text': 'example'}]}
    for role in ('user', 'assistant', 'user', 'assistant')
]


def write_data(root, train_key='source__a', val_key='source__b', omit_images=False):
    """Build paired tar members with image bytes preceding conversation JSON."""
    buf = io.BytesIO()
    Image.new('RGB', (2, 2), (12, 34, 56)).save(buf, format='PNG')
    for split, key in [('train', train_key), ('val', val_key)]:
        with tarfile.open(root / f'{split}-shard-000000.tar', 'w') as archive:
            fields = [
                ('jpgs', pickle.dumps([buf.getvalue()])),
                ('json', json.dumps(MESSAGES).encode()),
            ]
            for suffix, data in fields:
                if omit_images and suffix == 'jpgs':
                    continue
                member = tarfile.TarInfo(key + '.' + suffix)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    manifest = {
        'revision': provider.REVISION,
        'source_integrity_verified': True,
        'counts': {'train': 1, 'val': 1},
        'subsets': ['source'],
    }
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def dataset(root, records, max_length=32):
    """Bind the deterministic processor to a real offset-reading Dataset."""
    ds = provider.NemotronDataset(
        root, records, target_length=10, max_length=max_length, vocab_size=64, image_token_id=9
    )
    ds._processor = Processor()
    return ds


def test_assistant_targets_are_shifted_and_special_tokens_are_excluded():
    """Assert the exact predicted token positions across two assistant turns."""
    sample = tokenize(Processor(), MESSAGES, [Image.new('RGB', (2, 2))])
    expected = [-100] * len(Processor.ids)
    expected[7], expected[8], expected[16] = 20, 21, 22
    assert sample['labels'].tolist() == expected
    assert sample['loss_mask'].nonzero().flatten().tolist() == [7, 8, 16]
    assert sample['input_ids'].tolist() == Processor.ids
    assert sample['pixel_values'].dtype == torch.bfloat16


def test_unclosed_assistant_turn_fails_instead_of_training_unbounded_text():
    """Reject a missing ChatML end delimiter before assigning targets."""
    p = Processor()
    p.ids = p.ids[:-1]
    with pytest.raises(ValueError, match='no closing'):
        tokenize(p, MESSAGES, [])


def test_tar_index_pairs_components_independent_of_order_and_repeats_within_split(tmp_path):
    """Resolve reversed tar components and repeat without crossing split boundaries."""
    write_data(tmp_path)
    index = provider.build_index(tmp_path)
    ds = dataset(tmp_path, index['splits']['train'])
    assert len(ds) == 10
    assert index['splits']['train'][0]['key'] == 'source__a'
    assert index['splits']['val'][0]['key'] == 'source__b'
    assert torch.equal(ds[0]['labels'], ds[9]['labels'])
    assert ds.__getstate__()['_files'] == {}
    assert ds.__getstate__()['_processor'] is None
    assert provider.build_index(tmp_path) == index


@pytest.mark.parametrize('problem', ['missing_images', 'split_overlap', 'count_mismatch'])
def test_invalid_prepared_pairing_and_split_counts_fail(tmp_path, problem):
    """Reject missing images, duplicate split keys and inconsistent counts."""
    manifest = write_data(
        tmp_path,
        val_key='source__a' if problem == 'split_overlap' else 'source__b',
        omit_images=problem == 'missing_images',
    )
    if problem == 'count_mismatch':
        manifest['counts']['train'] = 2
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Missing image|duplicate/split-overlap|index count'):
        provider.build_index(tmp_path)
    assert not (tmp_path / provider.INDEX).exists()


def test_existing_index_rejects_changed_provenance(tmp_path):
    """Reject reuse after the conversion manifest changes."""
    manifest = write_data(tmp_path)
    provider.build_index(tmp_path)
    manifest['subsets'] = ['different-source']
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='does not match'):
        provider.build_index(tmp_path)


@pytest.mark.parametrize('problem', ['over_budget', 'vocabulary', 'image_grid'])
def test_bad_training_record_fails_with_key_instead_of_silent_replacement(tmp_path, problem):
    """Report invalid token budgets, vocabulary and image grids with the source key."""
    write_data(tmp_path)
    ds = dataset(
        tmp_path,
        provider.build_index(tmp_path)['splits']['train'],
        max_length=16 if problem == 'over_budget' else 32,
    )
    if problem == 'vocabulary':
        ds.vocab_size = 30
    if problem == 'image_grid':
        ds.image_token_id = 8
    with pytest.raises(RuntimeError, match='Nemotron record source__a:'):
        ds[0]


def test_packing_probe_can_run_directly_from_the_checkout():
    """Check the direct-script import path used for the packing probe."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    result = subprocess.run(
        [
            sys.executable,
            str(root / 'examples/multimodal_dev/data/nemotron/probe_pr48.py'),
            '--help',
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '--token-budget' in result.stdout


def test_header_census_expands_only_image_tokens_and_rejects_count_mismatch():
    """Expand image placeholders while preserving all surrounding text tokens."""
    from examples.multimodal_dev.data.nemotron.assess_records import expand_visual_tokens

    assert expand_visual_tokens([1, 9, 2, 8, 9, 3], 9, [2, 3]) == [1, 9, 9, 2, 8, 9, 9, 9, 3]
    with pytest.raises(ValueError, match='placeholder count'):
        expand_visual_tokens([1, 9, 2], 9, [2, 3])


def selection_fixture():
    """Build a source census with an exactly fitting and an overlong record."""
    from examples.multimodal_dev.data.nemotron.prepare_selection import make_selection

    subset = 'sec'
    index = {
        'manifest': {'verified_source_files': [{'path': 'sec/sec.jsonl', 'sha256': 'source-hash'}]},
        'splits': {
            'train': [{'key': 'sec__a'}, {'key': 'sec__long'}, {'key': 'other__x'}],
            'val': [{'key': 'sec__b'}],
        },
    }
    summary = {
        'processor': provider.PROCESSOR,
        'processor_revision': provider.PROCESSOR_REVISION,
        'min_pixels': 4096,
        'max_pixels': 262144,
        'subsets': {subset: {'source_sha256': 'source-hash'}},
    }
    lengths = [
        {'key': 'sec__a', 'tokens': 32},
        {'key': 'sec__long', 'tokens': 33},
        {'key': 'sec__b', 'tokens': 20},
    ]
    selection = make_selection(index, summary, lengths, subset=subset, token_budget=32)
    return index, selection


def test_complete_record_selection_preserves_budget_boundary_and_split():
    """Keep the boundary record and exclude the entire overlong record."""
    index, selection = selection_fixture()
    chosen = provider.select_records(index, selection, 32)
    assert chosen == {'train': [{'key': 'sec__a'}], 'val': [{'key': 'sec__b'}]}
    assert selection['counts']['train'] == {
        'source_records': 2,
        'selected_records': 1,
        'excluded_over_budget': 1,
    }
    assert index['splits']['train'][1]['key'] == 'sec__long'  # original index retained


@pytest.mark.parametrize(
    'problem', ['budget', 'provenance', 'processor', 'duplicate', 'cross_split', 'empty']
)
def test_selection_rejects_stale_or_invalid_training_views(problem):
    """Reject selection metadata and keys that disagree with training inputs."""
    index, selection = selection_fixture()
    if problem == 'budget':
        selection['token_budget'] = 64
    if problem == 'provenance':
        selection['manifest'] = {}
    if problem == 'processor':
        selection['max_pixels'] = 65536
    if problem == 'duplicate':
        selection['splits']['train'] *= 2
    if problem == 'cross_split':
        selection['splits']['train'] = ['sec__b']
    if problem == 'empty':
        selection['splits']['val'] = []
    with pytest.raises(ValueError):
        provider.select_records(index, selection, 32)


def test_census_accepts_string_answers_and_typed_user_images():
    """Normalize source messages without changing their text or image order."""
    from examples.multimodal_dev.data.nemotron.assess_records import normalize_source_messages

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': 'page.png'},
                {'type': 'text', 'text': 'question'},
            ],
        },
        {'role': 'assistant', 'content': 'answer'},
        {'role': 'user', 'content': ['another question']},
    ]
    normalized, refs = normalize_source_messages(messages)
    assert refs == ['page.png']
    assert normalized[0]['content'][0] == {'type': 'image'}
    assert normalized[1]['content'] == [{'type': 'text', 'text': 'answer'}]
    assert normalized[2]['content'] == [{'type': 'text', 'text': 'another question'}]
    assert messages[1]['content'] == 'answer'


@pytest.mark.parametrize('subset', ['clevr_2', 'long_document_sec_2'])
def test_all_fitting_subsets_still_require_an_explicit_selection(tmp_path, subset):
    """Require admission evidence for both families even when every record fits."""
    from examples.multimodal_dev.data.nemotron.prepare_selection import make_selection

    manifest = write_data(tmp_path, train_key=f'{subset}__a', val_key=f'{subset}__b')
    manifest.update(
        subsets=[subset],
        verified_source_files=[{'path': f'{subset}/{subset}.jsonl', 'sha256': 'source-hash'}],
    )
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    index = provider.build_index(tmp_path)
    with pytest.raises(ValueError, match='selection JSON, not a tar directory'):
        provider.load_training_selection(tmp_path, 32)
    summary = {
        'processor': provider.PROCESSOR,
        'processor_revision': provider.PROCESSOR_REVISION,
        'min_pixels': 4096,
        'max_pixels': 262144,
        'subsets': {subset: {'source_sha256': 'source-hash'}},
    }
    lengths = [{'key': f'{subset}__a', 'tokens': 32}, {'key': f'{subset}__b', 'tokens': 20}]
    selection = make_selection(index, summary, lengths, subset=subset, token_budget=32)
    assert all(
        c['selected_records'] == c['source_records'] == 1 and c['excluded_over_budget'] == 0
        for c in selection['counts'].values()
    )
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(selection))
    root, records = provider.load_training_selection(path, 32)
    assert root == tmp_path and records == index['splits']
    with pytest.raises(ValueError, match='budget differs'):
        provider.load_training_selection(path, 64)
    with pytest.raises(ValueError, match='exactly once'):
        make_selection(index, summary, lengths[:1], subset=subset, token_budget=32)


@pytest.mark.parametrize('subset', ['clevr_2', 'long_document_sec_2'])
def test_uniform_selection_excludes_whole_overlength_records(subset):
    """Apply identical exact-boundary admission to SEC and CLEVR."""
    from examples.multimodal_dev.data.nemotron.prepare_selection import make_selection

    index = {
        'manifest': {
            'verified_source_files': [{'path': f'{subset}/{subset}.jsonl', 'sha256': 'hash'}]
        },
        'splits': {
            'train': [{'key': f'{subset}__ok'}, {'key': f'{subset}__long'}],
            'val': [{'key': f'{subset}__val'}],
        },
    }
    summary = {
        'processor': provider.PROCESSOR,
        'processor_revision': provider.PROCESSOR_REVISION,
        'min_pixels': 4096,
        'max_pixels': 262144,
        'subsets': {subset: {'source_sha256': 'hash'}},
    }
    lengths = [
        {'key': f'{subset}__ok', 'tokens': 32},
        {'key': f'{subset}__long', 'tokens': 33},
        {'key': f'{subset}__val', 'tokens': 12},
    ]
    selection = make_selection(index, summary, lengths, subset=subset, token_budget=32)
    assert selection['splits'] == {'train': [f'{subset}__ok'], 'val': [f'{subset}__val']}
    assert selection['counts']['train']['excluded_over_budget'] == 1
    assert lengths[1]['tokens'] == 33


def test_selection_publication_is_immutable_and_idempotent(tmp_path):
    """Preserve existing selection bytes and mtime across repeated preparation."""
    from examples.multimodal_dev.data.nemotron.prepare_selection import publish_selection

    path = tmp_path / 'selection.json'
    value = {'token_budget': 32, 'splits': {'train': ['sample']}}
    publish_selection(path, value)
    before = path.stat().st_mtime_ns
    assert json.loads(path.read_text()) == value
    assert not path.with_suffix('.json.tmp').exists()
    publish_selection(path, value)
    assert path.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match='Existing selection differs'):
        publish_selection(path, dict(value, token_budget=64))
    assert json.loads(path.read_text()) == value
