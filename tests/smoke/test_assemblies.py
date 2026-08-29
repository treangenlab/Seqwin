import gzip
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import seqwin.assemblies as assemblies_module
from seqwin.assemblies import Assemblies, get_assemblies


class FakePopen:
    """Minimal makeblastdb process that captures standard input."""
    instances = []
    returncode_to_use = 0

    def __init__(self, args, stdin, stdout, stderr, text) -> None:
        self.args = args
        self.stdin = io.BytesIO()
        self.returncode = None
        self._final_returncode = self.returncode_to_use
        self.terminated = False
        self.instances.append(self)

    def communicate(self):
        if self.returncode is None:
            self.returncode = self._final_returncode
        return b'makeblastdb stdout', b'makeblastdb stderr'

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15


@pytest.fixture
def fake_popen(monkeypatch):
    FakePopen.instances = []
    FakePopen.returncode_to_use = 0
    monkeypatch.setattr(assemblies_module.subprocess, 'Popen', FakePopen)
    return FakePopen


def test_assemblies_is_a_lightweight_ordered_container(tmp_path: Path) -> None:
    paths = [
        tmp_path / 'negative-0.fasta', tmp_path / 'target-0.fasta',
        tmp_path / 'negative-1.fasta', tmp_path / 'target-1.fasta'
    ]
    is_targets = [False, True, False, True]

    assemblies = Assemblies(paths, is_targets)

    assert not isinstance(assemblies, pd.DataFrame)
    assert assemblies.paths == tuple(paths)
    np.testing.assert_array_equal(
        assemblies.is_targets, is_targets
    )
    assert assemblies.is_targets.dtype == np.bool_
    assert assemblies.is_targets.ndim == 1
    assert assemblies.is_targets.flags.c_contiguous
    assert len(assemblies) == 4
    assert not hasattr(assemblies, 'record_ids')
    assert not hasattr(assemblies, 'empty')
    assert not hasattr(assemblies, 'index')
    assert not hasattr(assemblies, 'to_csv')


def test_assemblies_empty_and_pickle_round_trip() -> None:
    assemblies = pickle.loads(pickle.dumps(Assemblies([], [])))

    assert len(assemblies) == 0
    assert assemblies.paths == ()
    assert assemblies.is_targets.dtype == np.bool_
    assert assemblies.is_targets.flags.c_contiguous


def test_assemblies_rejects_invalid_target_statuses(tmp_path: Path) -> None:
    paths = [tmp_path / 'first.fasta', tmp_path / 'second.fasta']

    with np.testing.assert_raises_regex(
        ValueError, r'len\(paths\) must equal len\(is_targets\)'
    ):
        Assemblies(paths, [True])
    with np.testing.assert_raises_regex(
        ValueError, 'is_targets must be one-dimensional'
    ):
        Assemblies(paths, [[True], [False]])


def test_get_assemblies_keeps_targets_first(monkeypatch, tmp_path: Path) -> None:
    tar_paths = [tmp_path / 'target-0.fasta', tmp_path / 'target-1.fasta']
    neg_paths = [tmp_path / 'negative-0.fasta', tmp_path / 'negative-1.fasta']
    config = type('Config', (), {
        'tar_paths': None, 'neg_paths': None, 'tar_dir': None, 'neg_dir': None,
        'overwrite': False, 'download_only': False
    })()
    state = type('RunState', (), {'working_dir': tmp_path})()
    monkeypatch.setattr(
        assemblies_module, '_download', lambda config, working_dir: (tar_paths, neg_paths)
    )

    assemblies = get_assemblies(config, state)

    assert assemblies.paths == tuple(tar_paths + neg_paths)
    np.testing.assert_array_equal(
        assemblies.is_targets, [True, True, False, False]
    )


def test_fetch_seq_preserves_multiindex_values_and_sorting(
    tmp_path: Path,
) -> None:
    target = tmp_path / 'target.fasta'
    negative = tmp_path / 'negative.fasta'
    target.write_text('>record-0\nAACCGGTT\n>record-1\nTTGGAACC\n')
    negative.write_text('>record-0\nGATTACA\n')
    assemblies = Assemblies([target, negative], [True, False])
    index = pd.MultiIndex.from_tuples(
        [('z', 2), ('a', 1), ('z', 0), ('a', 0)], names=['sample', 'interval']
    )
    locations = pd.DataFrame(
        {
            'assembly_idx': [1, 0, 1, 0],
            'record_idx': [0, 1, 0, 0],
            'start': [1, 2, 0, 0],
            'stop': [5, 6, 3, 4],
        },
        index=index,
    )

    actual = assemblies.fetch_seq(locations, n_cpu=1)
    expected = pd.Series(
        ['AACC', 'GGAA', 'GAT', 'ATTA'],
        index=pd.MultiIndex.from_tuples(
            [('a', 0), ('a', 1), ('z', 0), ('z', 2)],
            names=index.names,
        ),
    )

    pd.testing.assert_series_equal(actual, expected)


def test_fetch_seq_multiprocessing_matches_single_process(tmp_path: Path) -> None:
    first = tmp_path / 'first.fasta'
    second = tmp_path / 'second.fasta'
    first.write_text('>r0\nAACCGG\n')
    second.write_text('>r0\nTTGGCC\n')
    assemblies = Assemblies([first, second], [True, False])
    locations = pd.DataFrame({
        'assembly_idx': [1, 0],
        'record_idx': [0, 0],
        'start': [1, 2],
        'stop': [5, 6],
    }, index=pd.Index([5, 2], name='location'))

    expected = assemblies.fetch_seq(locations, n_cpu=1)
    actual = assemblies.fetch_seq(locations, n_cpu=8)

    pd.testing.assert_series_equal(actual, expected)


def test_fetch_seq_empty_preserves_index() -> None:
    assemblies = Assemblies([], [])
    index = pd.MultiIndex.from_arrays([[], []], names=['sample', 'interval'])
    locations = pd.DataFrame(
        columns=['assembly_idx', 'record_idx', 'start', 'stop'], index=index
    )

    actual = assemblies.fetch_seq(locations, n_cpu=4)

    assert actual.empty
    assert actual.dtype == object
    assert actual.index.equals(index)


@pytest.mark.parametrize('n_cpu', [1, 3])
def test_makeblastdb_streams_plain_and_gzip_fasta_in_order(
    fake_popen, tmp_path: Path, n_cpu: int,
) -> None:
    first = tmp_path / 'first.fasta'
    second = tmp_path / 'second.fasta.gz'
    third = tmp_path / 'third.fasta'
    first.write_bytes(b'>first one\nAAC\n>first two description\nGGT\n')
    second.write_bytes(gzip.compress(b'>second original header\nTTT\n'))
    third.write_bytes(b'>third header\nCCC\n')
    assemblies = Assemblies([first, second, third], [True, False, True])

    blastdb = assemblies.makeblastdb(
        tmp_path / f'db-{n_cpu}', neg_only=False, overwrite=False, n_cpu=n_cpu
    )

    assert blastdb.name == assemblies_module.BLASTCONFIG.title_all
    assert fake_popen.instances[-1].stdin.getvalue() == (
        b'>0@y@first one\nAAC\n>0@y@first two description\nGGT\n'
        b'>1@n@second original header\nTTT\n'
        b'>2@y@third header\nCCC\n'
    )


def test_makeblastdb_neg_only_preserves_global_indices(fake_popen, tmp_path: Path) -> None:
    paths = []
    for idx in range(4):
        path = tmp_path / f'{idx}.fasta'
        path.write_bytes(f'>record {idx}\nACGT\n'.encode())
        paths.append(path)
    assemblies = Assemblies(paths, [True, False, True, False])

    assemblies.makeblastdb(
        tmp_path / 'negative-db', neg_only=True, overwrite=False, n_cpu=3
    )

    assert fake_popen.instances[-1].stdin.getvalue() == (
        b'>1@n@record 1\nACGT\n>3@n@record 3\nACGT\n'
    )


def test_makeblastdb_worker_exception_propagates_and_reaps_process(
    fake_popen, tmp_path: Path,
) -> None:
    missing = tmp_path / 'missing.fasta'
    assemblies = Assemblies([missing], [False])

    with pytest.raises(FileNotFoundError):
        assemblies.makeblastdb(
            tmp_path / 'failed-db', neg_only=False, overwrite=False, n_cpu=2
        )

    proc = fake_popen.instances[-1]
    assert proc.terminated
    assert proc.returncode == -15


def test_makeblastdb_nonzero_return_writes_log_and_raises(fake_popen, tmp_path: Path) -> None:
    fasta = tmp_path / 'input.fasta'
    fasta.write_bytes(b'>record\nACGT\n')
    assemblies = Assemblies([fasta], [True])
    fake_popen.returncode_to_use = 2
    prefix = tmp_path / 'failed-db'

    with pytest.raises(RuntimeError, match='Failed to create the BLAST database'):
        assemblies.makeblastdb(prefix, neg_only=False, overwrite=False, n_cpu=1)

    assert (prefix / assemblies_module.WORKINGDIR.blast_log).read_text() == '\n'.join((
        str(fake_popen.instances[-1].args),
        'makeblastdb stdout',
        'makeblastdb stderr'
    ))
