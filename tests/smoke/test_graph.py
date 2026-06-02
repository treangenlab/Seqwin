from pathlib import Path

import networkx as nx
import numpy as np

from seqwin.graph import KMER_DTYPE, NODE_DTYPE, EDGE_DTYPE, build, _filter_kmers
from seqwin.markers import _create_ck


def _sorted_edges(edges: np.ndarray) -> np.ndarray:
    edge_values = edges.view(np.uint64).reshape(-1, 3)
    idx = np.lexsort((edge_values[:, 2], edge_values[:, 1], edge_values[:, 0]))
    return edge_values[idx]


def _assert_idx_invariants(kmers: np.ndarray, idx: np.ndarray, nodes: np.ndarray) -> None:
    assert idx.dtype == np.uintp
    assert len(idx) == len(kmers)
    assert np.array_equal(np.sort(idx), np.arange(len(kmers), dtype=np.uintp))
    for node in nodes:
        start = int(node['start'])
        stop = int(node['stop'])
        assert stop > start
        block = kmers[start:stop]
        assert len(block) == (stop - start)
        node_idx = idx[start:stop]
        assert len(node_idx) == len(block)
        assert not np.array_equal(node_idx, np.arange(start, stop, dtype=np.uintp))


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
    is_targets = [True, True, False, False]

    kmers_1, idx_1, nodes_1, edges_1, record_offsets_1, record_ids_1 = build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        is_targets=is_targets,
        n_cpu=1,
    )
    kmers_2, idx_2, nodes_2, edges_2, record_offsets_2, record_ids_2 = build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        is_targets=is_targets,
        n_cpu=2,
    )
    kmers_many, idx_many, nodes_many, edges_many, record_offsets_many, record_ids_many = build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        is_targets=is_targets,
        n_cpu=99,
    )

    assert kmers_1.dtype == KMER_DTYPE
    assert kmers_1.dtype.itemsize == 8
    assert kmers_1.dtype.names == ('pos', 'record_idx')
    for node in nodes_1:
        start = int(node['start'])
        stop = int(node['stop'])
        assert stop > start
        assert len(kmers_1[start:stop]) == (stop - start)

    assert np.array_equal(record_offsets_1, np.array([0, 1, 2, 3, 4], dtype=np.uintp))
    assert np.array_equal(np.unique(kmers_1['record_idx']), np.arange(4, dtype=np.uint32))
    assert np.all(nodes_1['n_tar'] + nodes_1['n_neg'] > 0)

    assert edges_1.ndim == 1
    assert edges_1.dtype == EDGE_DTYPE
    edge_values_1 = edges_1.view(np.uint64).reshape(-1, 3)
    assert np.array_equal(edge_values_1[:, 0], edges_1["first"])
    assert np.array_equal(edge_values_1[:, 1], edges_1["second"])
    assert np.array_equal(edge_values_1[:, 2], edges_1["weight"])
    assert nodes_1.dtype == NODE_DTYPE

    _assert_idx_invariants(kmers_1, idx_1, nodes_1)
    _assert_idx_invariants(kmers_2, idx_2, nodes_2)
    _assert_idx_invariants(kmers_many, idx_many, nodes_many)

    assert np.array_equal(kmers_1, kmers_2)
    assert np.array_equal(kmers_1, kmers_many)
    assert np.array_equal(idx_1, idx_2)
    assert np.array_equal(idx_1, idx_many)
    assert np.array_equal(nodes_1, nodes_2)
    assert np.array_equal(nodes_1, nodes_many)

    assert record_ids_1 == record_ids_2
    assert record_ids_1 == record_ids_many
    assert len(record_ids_1) == len(assembly_paths)
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

    kmers, _, _, _, record_offsets, record_ids = build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        is_targets=[True, True, False, False],
        n_cpu=2,
    )

    assert [len(ids) for ids in record_ids] == [2, 1, 3, 1]
    assert np.array_equal(record_offsets, np.array([0, 2, 3, 6, 7], dtype=np.uintp))
    assert np.array_equal(np.unique(kmers['record_idx']), np.arange(7, dtype=np.uint32))


def test_create_ck_reconstructs_assembly_and_local_record_indices(monkeypatch) -> None:
    captured = {}

    class FakeConnectedKmers:
        def __init__(self, graph, kmers_df, kmerlen):
            captured['graph'] = graph
            captured['kmers_df'] = kmers_df.copy()
            captured['kmerlen'] = kmerlen

    monkeypatch.setattr('seqwin.markers.ConnectedKmers', FakeConnectedKmers)

    graph = nx.Graph()
    kmers = (
        np.array([(5, 0), (6, 2), (7, 5)], dtype=KMER_DTYPE),
    )
    idx = (np.array([10, 20, 30], dtype=np.uintp),)
    record_offsets = np.array([0, 2, 5, 6], dtype=np.uintp)

    _create_ck(graph, (np.uint64(123),), kmers, idx, record_offsets, n_tar=2, kmerlen=7)

    df = captured['kmers_df'].sort_index()
    assert df['assembly_idx'].tolist() == [0, 1, 2]
    assert df['record_idx'].tolist() == [0, 0, 0]
    assert df['is_target'].tolist() == [True, True, False]
    assert df['hash'].tolist() == [123, 123, 123]
    assert captured['kmerlen'] == 7


def test_filter_kmers() -> None:
    kmers = np.array([
        (10, 0),
        (11, 0),
        (20, 1),
        (30, 2),
        (31, 2),
        (32, 2),
    ], dtype=KMER_DTYPE)
    idx = np.array([100, 101, 200, 300, 301, 302], dtype=np.uintp)
    nodes = np.array([
        (10, 0, 2, 1, 0, 0.1),
        (20, 2, 3, 1, 0, 0.2),
        (30, 3, 6, 1, 1, 0.3),
    ], dtype=NODE_DTYPE)

    kmers_new, idx_new, nodes_new = _filter_kmers(kmers, idx, nodes, {30, 10})

    assert np.array_equal(nodes_new['hash'], np.array([10, 30], dtype=np.uint64))
    assert np.array_equal(nodes_new['start'], np.array([0, 2], dtype=np.uintp))
    assert np.array_equal(nodes_new['stop'], np.array([2, 5], dtype=np.uintp))

    expected_kmers = np.array([
        (10, 0),
        (11, 0),
        (30, 2),
        (31, 2),
        (32, 2),
    ], dtype=KMER_DTYPE)
    expected_idx = np.array([100, 101, 300, 301, 302], dtype=np.uintp)

    assert np.array_equal(kmers_new, expected_kmers)
    assert np.array_equal(idx_new, expected_idx)
