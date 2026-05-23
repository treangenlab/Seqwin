import numpy as np

from seqwin.graph import KMER_DTYPE, NODE_DTYPE, build, _filter_kmers


def _sorted_edges(edges: np.ndarray) -> np.ndarray:
    idx = np.lexsort((edges[:, 2], edges[:, 1], edges[:, 0]))
    return edges[idx]


def _assert_idx_invariants(kmers: np.ndarray, idx: np.ndarray, nodes: np.ndarray) -> None:
    assert idx.dtype == np.uint64
    assert len(idx) == len(kmers)
    assert np.array_equal(np.sort(idx), np.arange(len(kmers), dtype=np.uint64))
    for node in nodes:
        start = int(node['start'])
        stop = int(node['stop'])
        assert stop > start
        block = kmers[start:stop]
        assert len(block) == (stop - start)
        node_idx = idx[start:stop]
        assert len(node_idx) == len(block)
        assert not np.array_equal(node_idx, np.arange(start, stop, dtype=np.uint64))


def test_dtype_layouts() -> None:
    assert KMER_DTYPE.itemsize == 8
    assert KMER_DTYPE.names == ('pos', 'record_idx', 'assembly_idx')
    assert KMER_DTYPE['assembly_idx'] == np.dtype(np.uint16)

    assert NODE_DTYPE.names == ('hash', 'start', 'stop', 'n_tar', 'n_neg', 'penalty')
    assert NODE_DTYPE["n_tar"] == np.dtype(np.uint32)
    assert NODE_DTYPE["n_neg"] == np.dtype(np.uint32)
    assert NODE_DTYPE.itemsize == 40


def test_build_threading_equivalence(targets_dir, non_targets_dir) -> None:
    assembly_paths = [
        targets_dir / 'target-1.fasta',
        targets_dir / 'target-2.fasta',
        non_targets_dir / 'non-target-1.fasta',
        non_targets_dir / 'non-target-2.fasta',
    ]
    assembly_idx = [0, 1, 2, 3]
    is_target = [True, True, False, False]

    kmers_1, idx_1, nodes_1, edges_1, record_ids_1 = build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        assembly_idx=assembly_idx,
        is_target=is_target,
        n_cpu=1,
    )
    kmers_2, idx_2, nodes_2, edges_2, record_ids_2 = build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        assembly_idx=assembly_idx,
        is_target=is_target,
        n_cpu=2,
    )
    kmers_many, idx_many, nodes_many, edges_many, record_ids_many = build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        assembly_idx=assembly_idx,
        is_target=is_target,
        n_cpu=99,
    )

    assert kmers_1.dtype == KMER_DTYPE
    assert kmers_1.dtype.itemsize == 8
    assert kmers_1.dtype.names == ('pos', 'record_idx', 'assembly_idx')
    for node in nodes_1:
        start = int(node['start'])
        stop = int(node['stop'])
        assert stop > start
        assert len(kmers_1[start:stop]) == (stop - start)

    assert set(np.unique(kmers_1['assembly_idx']).tolist()) == {0, 1, 2, 3}
    assert np.all(nodes_1['n_tar'] + nodes_1['n_neg'] > 0)

    assert edges_1.shape[1] == 3
    assert edges_1.dtype == np.uint64
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

    assert np.array_equal(_sorted_edges(edges_1), _sorted_edges(edges_2))
    assert np.array_equal(_sorted_edges(edges_1), _sorted_edges(edges_many))


def test_filter_kmers() -> None:
    kmers = np.array([
        (10, 0, 0),
        (11, 0, 0),
        (20, 0, 1),
        (30, 1, 1),
        (31, 1, 1),
        (32, 1, 1),
    ], dtype=KMER_DTYPE)
    idx = np.array([100, 101, 200, 300, 301, 302], dtype=np.uint64)
    nodes = np.array([
        (10, 0, 2, 1, 0, 0.1),
        (20, 2, 3, 1, 0, 0.2),
        (30, 3, 6, 1, 1, 0.3),
    ], dtype=NODE_DTYPE)

    kmers_new, idx_new, nodes_new = _filter_kmers(kmers, idx, nodes, {30, 10})

    assert np.array_equal(nodes_new['hash'], np.array([10, 30], dtype=np.uint64))
    assert np.array_equal(nodes_new['start'], np.array([0, 2], dtype=np.uint64))
    assert np.array_equal(nodes_new['stop'], np.array([2, 5], dtype=np.uint64))

    expected_kmers = np.array([
        (10, 0, 0),
        (11, 0, 0),
        (30, 1, 1),
        (31, 1, 1),
        (32, 1, 1),
    ], dtype=KMER_DTYPE)
    expected_idx = np.array([100, 101, 300, 301, 302], dtype=np.uint64)

    assert np.array_equal(kmers_new, expected_kmers)
    assert np.array_equal(idx_new, expected_idx)
