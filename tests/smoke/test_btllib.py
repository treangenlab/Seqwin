import numpy as np

from seqwin.btllib import indexlr


def _sorted_edges(edges: np.ndarray) -> np.ndarray:
    idx = np.lexsort((edges[:, 2], edges[:, 1], edges[:, 0]))
    return edges[idx]


def test_indexlr_threading_equivalence(targets_dir, non_targets_dir) -> None:
    assembly_paths = [
        targets_dir / 'target-1.fasta',
        targets_dir / 'target-2.fasta',
        non_targets_dir / 'non-target-1.fasta',
        non_targets_dir / 'non-target-2.fasta',
    ]
    assembly_idx = [0, 1, 2, 3]
    is_target = [True, True, False, False]

    kmers_1, edges_1, record_ids_1 = indexlr(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        assembly_idx=assembly_idx,
        is_target=is_target,
        n_cpu=1,
    )
    kmers_2, edges_2, record_ids_2 = indexlr(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        assembly_idx=assembly_idx,
        is_target=is_target,
        n_cpu=2,
    )
    kmers_many, edges_many, record_ids_many = indexlr(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        assembly_idx=assembly_idx,
        is_target=is_target,
        n_cpu=99,
    )

    assert edges_1.shape[1] == 3
    assert edges_1.dtype == np.uint64
    assert np.array_equal(kmers_1, kmers_2)
    assert np.array_equal(kmers_1, kmers_many)

    assert record_ids_1 == record_ids_2
    assert record_ids_1 == record_ids_many
    assert len(record_ids_1) == len(assembly_paths)

    assert np.array_equal(_sorted_edges(edges_1), _sorted_edges(edges_2))
    assert np.array_equal(_sorted_edges(edges_1), _sorted_edges(edges_many))
