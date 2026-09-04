from pathlib import Path

import numpy as np
import pytest

from seqwin.graph import KMER_DTYPE, NODE_DTYPE, EDGE_DTYPE, KmerGraph


def _build(*args, **kwargs):
    graph = KmerGraph(*args, **kwargs)
    return graph.kmers, graph.nodes, graph.edges, graph.record_offsets, graph.record_ids


def _small_graph(targets_dir: Path, non_targets_dir: Path) -> KmerGraph:
    return KmerGraph(
        [targets_dir / 'target-1.fasta', non_targets_dir / 'non-target-1.fasta'],
        kmerlen=7,
        windowsize=10,
    )


def test_graph_save_load_round_trip(tmp_path: Path, targets_dir: Path, non_targets_dir: Path) -> None:
    graph = _small_graph(targets_dir, non_targets_dir)
    graph_path = tmp_path / 'graph'
    graph_path.mkdir()
    graph.save(graph_path)

    names = ('kmers', 'nodes', 'edges', 'record_offsets', 'record_ids')
    assert {path.name for path in graph_path.iterdir()} == {f'{name}.npy' for name in names}

    loaded = KmerGraph.load(graph_path)
    for name in names:
        original_array = getattr(graph, name)
        loaded_array = getattr(loaded, name)
        assert loaded_array.dtype == original_array.dtype
        assert loaded_array.shape == original_array.shape
        assert np.array_equal(loaded_array, original_array)
        assert isinstance(loaded_array, np.memmap)
        assert loaded_array.flags.c_contiguous
        expected_mode = 'c' if name == 'nodes' else 'r'
        assert loaded_array.mode == expected_mode
        assert loaded_array.flags.writeable is (name == 'nodes')
    assert loaded.record_ids.dtype.kind == 'U'


@pytest.mark.parametrize(
    ('name', 'array', 'message'),
    (
        ('kmers', np.empty((1, 1), dtype=KMER_DTYPE), 'one-dimensional'),
        ('nodes', np.empty(1, dtype=KMER_DTYPE), 'has dtype'),
        ('record_ids', np.array(['record'], dtype='S6'), 'has dtype'),
    ),
)
def test_graph_load_rejects_malformed_arrays(
    tmp_path: Path,
    targets_dir: Path,
    non_targets_dir: Path,
    name: str,
    array: np.ndarray,
    message: str,
) -> None:
    graph_path = tmp_path / 'graph'
    graph_path.mkdir()
    _small_graph(targets_dir, non_targets_dir).save(graph_path)
    np.save(graph_path / f'{name}.npy', array)

    with pytest.raises(ValueError, match=message):
        KmerGraph.load(graph_path)


def test_graph_load_rejects_missing_array(tmp_path: Path, targets_dir: Path, non_targets_dir: Path) -> None:
    graph_path = tmp_path / 'graph'
    graph_path.mkdir()
    _small_graph(targets_dir, non_targets_dir).save(graph_path)
    (graph_path / 'edges.npy').unlink()

    with pytest.raises(FileNotFoundError, match='edges.npy'):
        KmerGraph.load(graph_path)


def _sorted_edges(edges: np.ndarray) -> np.ndarray:
    edge_values = edges.view(np.uint64).reshape(-1, 3)
    idx = np.lexsort((edge_values[:, 2], edge_values[:, 1], edge_values[:, 0]))
    return edge_values[idx]


def _assert_graph_outputs_equal(standard, low_memory) -> None:
    kmers_std, nodes_std, edges_std, offsets_std, ids_std = standard
    kmers_lm, nodes_lm, edges_lm, offsets_lm, ids_lm = low_memory

    assert np.array_equal(kmers_std, kmers_lm)
    assert np.array_equal(nodes_std, nodes_lm)
    assert np.array_equal(_sorted_edges(edges_std), _sorted_edges(edges_lm))
    assert offsets_std.dtype == np.dtype(np.uint32)
    assert offsets_lm.dtype == np.dtype(np.uint32)
    assert np.array_equal(offsets_std, offsets_lm)
    assert np.array_equal(ids_std, ids_lm)
    _assert_node_ranges(kmers_lm, nodes_lm)


def _assert_node_ranges(kmers: np.ndarray, nodes: np.ndarray) -> None:
    total = 0
    for node in nodes:
        start = int(node['start'])
        stop = int(node['stop'])
        assert 0 <= start <= stop <= len(kmers)
        assert len(kmers[start:stop]) == (stop - start)
        assert np.all(kmers[start:stop]['record_idx'][:-1] <= kmers[start:stop]['record_idx'][1:])
        total += stop - start
    assert total == len(kmers)


def test_dtype_layouts() -> None:
    assert KMER_DTYPE.itemsize == 8
    assert KMER_DTYPE.names == ('pos', 'record_idx')
    assert KMER_DTYPE['record_idx'] == np.dtype(np.uint32)

    assert np.dtype(np.uintp).itemsize == 8

    assert NODE_DTYPE.names == ('hash', 'start', 'stop', 'n_tar', 'n_neg', 'penalty')
    assert NODE_DTYPE["start"] == np.dtype(np.uintp)
    assert NODE_DTYPE["stop"] == np.dtype(np.uintp)
    assert NODE_DTYPE["n_tar"] == np.dtype(np.uint32)
    assert NODE_DTYPE["n_neg"] == np.dtype(np.uint32)
    assert NODE_DTYPE.itemsize == 40

    assert EDGE_DTYPE.names == ("first", "second", "weight")
    assert EDGE_DTYPE["weight"] == np.dtype(np.uintp)
    assert EDGE_DTYPE.itemsize == 24
    assert EDGE_DTYPE.fields["first"][1] == 0
    assert EDGE_DTYPE.fields["second"][1] == 8
    assert EDGE_DTYPE.fields["weight"][1] == 16


def test_build_threading_equivalence(targets_dir, non_targets_dir) -> None:
    assembly_paths = [
        targets_dir / 'target-1.fasta',
        targets_dir / 'target-2.fasta',
        non_targets_dir / 'non-target-1.fasta',
        non_targets_dir / 'non-target-2.fasta',
    ]
    kmers_1, nodes_1, edges_1, record_offsets_1, record_ids_1 = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=1,
    )
    kmers_2, nodes_2, edges_2, record_offsets_2, record_ids_2 = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=2,
    )
    kmers_many, nodes_many, edges_many, record_offsets_many, record_ids_many = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=99,
    )

    assert all(array.ndim == 1 for array in (kmers_1, nodes_1, edges_1, record_offsets_1, record_ids_1))

    assert np.array_equal(kmers_1, kmers_2)
    assert np.array_equal(kmers_1, kmers_many)
    assert np.array_equal(nodes_1, nodes_2)
    assert np.array_equal(nodes_1, nodes_many)

    assert np.array_equal(record_ids_1, record_ids_2)
    assert np.array_equal(record_ids_1, record_ids_many)
    assert np.array_equal(record_offsets_1, record_offsets_2)
    assert np.array_equal(record_offsets_1, record_offsets_many)

    assert np.array_equal(_sorted_edges(edges_1), _sorted_edges(edges_2))
    assert np.array_equal(_sorted_edges(edges_1), _sorted_edges(edges_many))


def test_multi_thread_record_offsets_and_global_record_indices(tmp_path: Path) -> None:
    def write_fasta(path: Path, n_records: int) -> None:
        seq = 'ACGT' * 20
        path.write_text(''.join(f'>r{i}\n{seq}\n' for i in range(n_records)))

    assembly_paths = []
    for i, n_records in enumerate([2, 1, 3, 1]):
        path = tmp_path / f'a{i}.fasta'
        write_fasta(path, n_records)
        assembly_paths.append(path)

    kmers, _, _, record_offsets, record_ids = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=2,
    )

    assert record_offsets.dtype == np.dtype(np.uint32)
    assert np.array_equal(record_offsets, np.array([0, 2, 3, 6, 7], dtype=np.uint32))
    assert record_ids.dtype.kind == 'U'
    assert record_ids.tolist() == ['r0', 'r1', 'r0', 'r0', 'r1', 'r2', 'r0']
    assert [record_ids[record_offsets[i]:record_offsets[i + 1]].tolist()
            for i in range(4)] == [['r0', 'r1'], ['r0'], ['r0', 'r1', 'r2'], ['r0']]
    assert len(record_ids) == int(record_offsets[-1])
    assert np.array_equal(np.unique(kmers['record_idx']), np.arange(7, dtype=np.uint32))


def test_build_empty_record_offsets(tmp_path: Path) -> None:
    empty_assembly = tmp_path / 'empty.fasta'
    empty_assembly.write_text('')

    for assembly_paths, expected_offsets in (
        ([], np.array([0], dtype=np.uint32)),
        ([empty_assembly], np.array([0, 0], dtype=np.uint32)),
    ):
        for low_memory in (False, True):
            kmers, _, _, record_offsets, record_ids = _build(
                assembly_paths,
                kmerlen=7,
                windowsize=10,
                n_cpu=2,
                low_memory=low_memory,
            )
            assert len(kmers) == 0
            assert record_offsets.dtype == np.dtype(np.uint32)
            assert np.array_equal(record_offsets, expected_offsets)
            assert isinstance(record_ids, np.ndarray)
            assert record_ids.ndim == 1
            assert record_ids.dtype.kind == 'U'
            assert record_ids.size == 0
            assert len(record_ids) == int(record_offsets[-1])


def test_low_memory_build_matches_standard(targets_dir, non_targets_dir) -> None:
    assembly_paths = [
        targets_dir / 'target-1.fasta',
        targets_dir / 'target-2.fasta',
        non_targets_dir / 'non-target-1.fasta',
        non_targets_dir / 'non-target-2.fasta',
    ]
    standard = _build(
        assembly_paths, kmerlen=7, windowsize=10, n_cpu=2, low_memory=False,
    )
    low_memory = _build(
        assembly_paths, kmerlen=7, windowsize=10, n_cpu=2, low_memory=True,
    )

    _assert_graph_outputs_equal(standard, low_memory)
