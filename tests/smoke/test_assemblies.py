import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from seqwin.assemblies import Assemblies


def test_assemblies_is_a_lightweight_ordered_container(tmp_path: Path) -> None:
    tar_paths = [tmp_path / 'target-0.fasta', tmp_path / 'target-1.fasta']
    neg_paths = [tmp_path / 'negative-0.fasta', tmp_path / 'negative-1.fasta']

    assemblies = Assemblies(tar_paths, neg_paths)

    assert not isinstance(assemblies, pd.DataFrame)
    assert assemblies.path == tuple(tar_paths + neg_paths)
    np.testing.assert_array_equal(
        assemblies.is_target, [True, True, False, False]
    )
    assert assemblies.is_target.dtype == np.bool_
    assert assemblies.is_target.ndim == 1
    assert assemblies.is_target.flags.c_contiguous
    assert len(assemblies) == 4
    assert not hasattr(assemblies, 'record_ids')
    assert not hasattr(assemblies, 'empty')
    assert not hasattr(assemblies, 'index')
    assert not hasattr(assemblies, 'to_csv')


def test_assemblies_empty_and_pickle_round_trip() -> None:
    assemblies = pickle.loads(pickle.dumps(Assemblies([], [])))

    assert len(assemblies) == 0
    assert assemblies.path == ()
    assert assemblies.is_target.dtype == np.bool_
    assert assemblies.is_target.flags.c_contiguous


def test_fetch_seq_preserves_multiindex_values_and_sorting(
    tmp_path: Path,
) -> None:
    target = tmp_path / 'target.fasta'
    negative = tmp_path / 'negative.fasta'
    target.write_text('>record-0\nAACCGGTT\n>record-1\nTTGGAACC\n')
    negative.write_text('>record-0\nGATTACA\n')
    assemblies = Assemblies([target], [negative])
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
    assemblies = Assemblies([first], [second])
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
