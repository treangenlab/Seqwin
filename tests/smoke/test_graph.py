import numpy as np

from seqwin.graph import NODE_DTYPE, build


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
        assert np.all(block['hash'] == node['hash'])
        node_idx = idx[start:stop]
        assert len(node_idx) == len(block)
        assert not np.array_equal(node_idx, np.arange(start, stop, dtype=np.uint64))


def test_node_dtype_layout() -> None:
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
