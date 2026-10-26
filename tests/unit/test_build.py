from pathlib import Path

import numpy as np
import pytest

from seqwin.core import KMER_DTYPE, EDGE_DTYPE, Graph


def _build(*args, **kwargs):
    graph = Graph(*args, **kwargs)
    return (
        graph.kmers, graph.offsets, graph.prevalences, graph.hashes, graph.edges, graph.record_offsets,
        graph.assembly_nodes, graph.node_offsets,
    )


def _small_graph(targets_dir: Path, non_targets_dir: Path) -> Graph:
    return Graph(
        [targets_dir / 'target-1.fasta', non_targets_dir / 'non-target-1.fasta'],
        kmerlen=7,
        windowsize=10,
    )


def test_graph_save_load_round_trip(tmp_path: Path, targets_dir: Path, non_targets_dir: Path) -> None:
    graph = _small_graph(targets_dir, non_targets_dir)
    graph_path = tmp_path / 'graph'
    graph_path.mkdir()
    graph.save(graph_path)

    names = (
        'kmers', 'offsets', 'prevalences', 'hashes', 'edges', 'record_offsets',
        'assembly_nodes', 'node_offsets',
    )
    assert {path.name for path in graph_path.iterdir()} == {f'{name}.npy' for name in names}

    loaded = Graph.load(graph_path)
    for name in names:
        original_array = getattr(graph, name)
        loaded_array = getattr(loaded, name)
        assert loaded_array.dtype == original_array.dtype
        assert loaded_array.shape == original_array.shape
        assert np.array_equal(loaded_array, original_array)
        assert isinstance(loaded_array, np.memmap)
        assert loaded_array.flags.c_contiguous
        expected_mode = 'r'
        assert loaded_array.mode == expected_mode


@pytest.mark.parametrize(
    ('name', 'array', 'message'),
    (
        ('kmers', np.empty((1, 1), dtype=KMER_DTYPE), 'one-dimensional'),
        ('offsets', np.empty(1, dtype=KMER_DTYPE), 'has dtype'),
        ('prevalences', np.empty(1, dtype=np.uintp), 'has dtype'),
        ('hashes', np.empty(1, dtype=np.uint32), 'has dtype'),
        ('assembly_nodes', np.empty(1, dtype=np.uint32), 'has dtype'),
        ('node_offsets', np.empty(0, dtype=np.uintp), 'must not be empty'),
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
        Graph.load(graph_path)


def test_graph_load_rejects_missing_array(tmp_path: Path, targets_dir: Path, non_targets_dir: Path) -> None:
    graph_path = tmp_path / 'graph'
    graph_path.mkdir()
    _small_graph(targets_dir, non_targets_dir).save(graph_path)
    (graph_path / 'edges.npy').unlink()

    with pytest.raises(FileNotFoundError, match='edges.npy'):
        Graph.load(graph_path)


def _assert_edges_weight_sorted(edges: np.ndarray) -> None:
    assert np.all(edges['weight'][:-1] >= edges['weight'][1:])
    for weight in np.unique(edges['weight']):
        tied = edges[edges['weight'] == weight]
        endpoints = tied[['first', 'second']].tolist()
        assert endpoints == sorted(endpoints)


def _assert_indexed_edges(nodes: np.ndarray, edges: np.ndarray) -> None:
    assert np.all(edges['first'] < len(nodes))
    assert np.all(edges['second'] < len(nodes))
    assert np.all(edges['first'] <= edges['second'])


def _assert_graph_outputs_equal(standard, low_memory) -> None:
    for standard_array, low_memory_array in zip(standard, low_memory, strict=True):
        np.testing.assert_array_equal(standard_array, low_memory_array)
    kmers, offsets, prevalences, hashes, edges, *_ = low_memory
    _assert_edges_weight_sorted(edges)
    _assert_indexed_edges(hashes, edges)
    _assert_node_ranges(kmers, offsets, prevalences, hashes)


def _assert_node_ranges(kmers, offsets, prevalences, hashes) -> None:
    assert kmers.dtype == KMER_DTYPE
    assert offsets.dtype == np.dtype(np.uintp)
    assert prevalences.dtype == np.dtype(np.uint32)
    assert hashes.dtype == np.dtype(np.uint64)
    assert len(offsets) == len(hashes) + 1
    assert len(prevalences) == len(hashes)
    assert offsets[0] == 0
    assert offsets[-1] == len(kmers)
    assert np.all(offsets[:-1] < offsets[1:])
    assert np.all(hashes[:-1] < hashes[1:])
    for start, stop in zip(offsets[:-1], offsets[1:]):
        records = kmers[start:stop]['record_idx']
        assert np.all(records[:-1] <= records[1:])


def test_dtype_layouts() -> None:
    assert KMER_DTYPE.itemsize == 8
    assert KMER_DTYPE.names == ('pos', 'record_idx')
    assert KMER_DTYPE['record_idx'] == np.dtype(np.uint32)

    assert np.dtype(np.uintp).itemsize == 8

    assert EDGE_DTYPE.names == ("first", "second", "weight")
    assert EDGE_DTYPE["first"] == np.dtype(np.uintp)
    assert EDGE_DTYPE["second"] == np.dtype(np.uintp)
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
    kmers_1, offsets_1, prevalences_1, hashes_1, edges_1, record_offsets_1, assembly_nodes_1, node_offsets_1 = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=1,
    )
    kmers_2, offsets_2, prevalences_2, hashes_2, edges_2, record_offsets_2, assembly_nodes_2, node_offsets_2 = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=2,
    )
    kmers_many, offsets_many, prevalences_many, hashes_many, edges_many, record_offsets_many, assembly_nodes_many, node_offsets_many = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=99,
    )

    assert all(array.ndim == 1 for array in (
        kmers_1, offsets_1, prevalences_1, hashes_1, edges_1, assembly_nodes_1, node_offsets_1,
        record_offsets_1,
    ))

    assert np.array_equal(kmers_1, kmers_2)
    assert np.array_equal(kmers_1, kmers_many)
    assert np.array_equal(offsets_1, offsets_2)
    assert np.array_equal(offsets_1, offsets_many)
    assert np.array_equal(prevalences_1, prevalences_2)
    assert np.array_equal(prevalences_1, prevalences_many)
    assert np.array_equal(hashes_1, hashes_2)
    assert np.array_equal(hashes_1, hashes_many)
    assert np.array_equal(assembly_nodes_1, assembly_nodes_2)
    assert np.array_equal(assembly_nodes_1, assembly_nodes_many)
    assert np.array_equal(node_offsets_1, node_offsets_2)
    assert np.array_equal(node_offsets_1, node_offsets_many)

    assert np.array_equal(record_offsets_1, record_offsets_2)
    assert np.array_equal(record_offsets_1, record_offsets_many)

    assert np.array_equal(edges_1, edges_2)
    assert np.array_equal(edges_1, edges_many)
    _assert_edges_weight_sorted(edges_1)
    _assert_edges_weight_sorted(edges_2)
    _assert_edges_weight_sorted(edges_many)
    _assert_indexed_edges(hashes_1, edges_1)
    _assert_node_ranges(kmers_1, offsets_1, prevalences_1, hashes_1)


def test_duplicate_edges_across_workers_are_aggregated(tmp_path: Path) -> None:
    assembly_paths = [tmp_path / f'assembly-{i}.fasta' for i in range(2)]
    for path in assembly_paths:
        path.write_text('>record\nACGTTGCATGTCGCATGATGCATGAGAGCT\n')

    single_worker = Graph(assembly_paths, kmerlen=5, windowsize=6, n_cpu=1)
    multiple_workers = Graph(assembly_paths, kmerlen=5, windowsize=6, n_cpu=2)

    assert len(multiple_workers.edges) > 0
    assert np.all(multiple_workers.edges['weight'] == 2)
    assert np.array_equal(multiple_workers.edges, single_worker.edges)
    _assert_edges_weight_sorted(multiple_workers.edges)
    _assert_indexed_edges(multiple_workers.hashes, multiple_workers.edges)


def test_multi_worker_record_offsets_and_global_record_indices(tmp_path: Path) -> None:
    def write_fasta(path: Path, n_records: int) -> None:
        seq = 'ACGT' * 20
        path.write_text(''.join(f'>r{i}\n{seq}\n' for i in range(n_records)))

    assembly_paths = []
    for i, n_records in enumerate([2, 1, 3, 1]):
        path = tmp_path / f'a{i}.fasta'
        write_fasta(path, n_records)
        assembly_paths.append(path)

    kmers, _, _, _, _, record_offsets, _, _ = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=2,
    )

    assert record_offsets.dtype == np.dtype(np.uint32)
    assert np.array_equal(record_offsets, np.array([0, 2, 3, 6, 7], dtype=np.uint32))
    assert np.array_equal(np.unique(kmers['record_idx']), np.arange(7, dtype=np.uint32))


def test_build_empty_record_offsets(tmp_path: Path) -> None:
    empty_assembly = tmp_path / 'empty.fasta'
    empty_assembly.write_text('')

    for assembly_paths, expected_offsets in (
        ([], np.array([0], dtype=np.uint32)),
        ([empty_assembly], np.array([0, 0], dtype=np.uint32)),
        ([empty_assembly] * 3, np.zeros(4, dtype=np.uint32)),
    ):
        for low_memory in (False, True):
            kmers, offsets, prevalences, hashes, _, record_offsets, assembly_nodes, node_offsets = _build(
                assembly_paths,
                kmerlen=7,
                windowsize=10,
                n_cpu=2,
                low_memory=low_memory,
            )
            assert len(kmers) == 0
            _assert_node_ranges(kmers, offsets, prevalences, hashes)
            np.testing.assert_array_equal(offsets, [0])
            assert len(hashes) == 0
            assert len(assembly_nodes) == 0
            assert assembly_nodes.dtype == np.dtype(np.uintp)
            assert np.array_equal(node_offsets, np.zeros(len(assembly_paths) + 1, dtype=np.uintp))
            assert record_offsets.dtype == np.dtype(np.uint32)
            assert np.array_equal(record_offsets, expected_offsets)


def test_low_memory_build_matches_standard(targets_dir, non_targets_dir) -> None:
    assembly_paths = [
        targets_dir / 'target-1.fasta',
        targets_dir / 'target-2.fasta',
        non_targets_dir / 'non-target-1.fasta',
        non_targets_dir / 'non-target-2.fasta',
    ]
    standard = _build(
        assembly_paths, kmerlen=7, windowsize=10, n_cpu=1, low_memory=False,
    )

    for n_cpu in (1, 2, 99):
        low_memory = _build(
            assembly_paths, kmerlen=7, windowsize=10, n_cpu=n_cpu, low_memory=True,
        )
        _assert_graph_outputs_equal(standard, low_memory)


def test_assembly_nodes_match_kmer_memberships(tmp_path: Path) -> None:
    assembly_paths = [tmp_path / f'assembly-{i}.fasta' for i in range(4)]
    assembly_paths[0].write_text('>r0\nACGTACGTACGTACGT\n>r1\nACGTACGTACGTACGT\n')
    assembly_paths[1].write_text('')
    assembly_paths[2].write_text('>short\nAC\n')
    assembly_paths[3].write_text('>r0\nTTTTACGTACGTGGGG\n')

    graph = Graph(assembly_paths, kmerlen=5, windowsize=6, n_cpu=3)
    assert graph.assembly_nodes.dtype == np.dtype(np.uintp)
    assert graph.node_offsets.dtype == np.dtype(np.uintp)
    assert len(graph.node_offsets) == len(assembly_paths) + 1
    assert graph.node_offsets[0] == 0
    assert graph.node_offsets[-1] == len(graph.assembly_nodes)
    assert np.all(graph.node_offsets[:-1] <= graph.node_offsets[1:])
    assert np.all(graph.assembly_nodes < len(graph.hashes))

    record_assembly = np.empty(int(graph.record_offsets[-1]), dtype=np.uint32)
    for assembly_i in range(len(assembly_paths)):
        record_assembly[
            graph.record_offsets[assembly_i]:graph.record_offsets[assembly_i + 1]
        ] = assembly_i

    expected = [[] for _ in assembly_paths]
    _assert_node_ranges(graph.kmers, graph.offsets, graph.prevalences, graph.hashes)
    for node_i in range(len(graph.hashes)):
        start, stop = graph.offsets[node_i:node_i + 2]
        assemblies = np.unique(record_assembly[
            graph.kmers['record_idx'][start:stop]
        ])
        assert graph.prevalences[node_i] == len(assemblies)
        for assembly_i in assemblies:
            expected[int(assembly_i)].append(node_i)

    for assembly_i, expected_nodes in enumerate(expected):
        actual = graph.assembly_nodes[
            graph.node_offsets[assembly_i]:graph.node_offsets[assembly_i + 1]
        ]
        assert np.array_equal(actual, np.asarray(expected_nodes, dtype=np.uintp))
        assert np.all(actual[:-1] < actual[1:])
        assert len(actual) == len(np.unique(actual))
