import numpy as np

import seqwin.markers as markers
from seqwin.graph import EDGE_DTYPE, KMER_DTYPE, NODE_DTYPE
from seqwin.kmers import FilteredGraph


def test_create_ck_uses_interleaved_target_mask(monkeypatch) -> None:
    captured = dict()

    def capture_connected_kmers(kmers, kmerlen, windowsize):
        captured['kmers'] = kmers
        return object()

    monkeypatch.setattr(markers, 'ConnectedKmers', capture_connected_kmers)
    kmers = np.array(
        [(10, 0), (20, 1), (30, 2), (40, 3)],
        dtype=KMER_DTYPE
    )
    is_targets = np.array([False, True, False, True], dtype=np.bool_)

    markers._create_ck(
        (np.uint64(1),),
        (kmers,),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
        is_targets,
        7,
        10
    )

    result = captured['kmers'].sort_values('assembly_idx')
    assert np.array_equal(result['assembly_idx'], np.arange(4))
    assert np.array_equal(result['is_target'], is_targets)


def test_create_ck_args_slices_non_contiguous_raw_kmer_ranges() -> None:
    kmers = np.array(
        [(10, 0), (11, 1), (99, 0), (20, 0), (21, 1)],
        dtype=KMER_DTYPE,
    )
    graph = FilteredGraph(
        kmers=kmers,
        nodes=np.array(
            [(10, 0, 2, 0, 0, 0.0), (20, 3, 5, 0, 0, 0.0)],
            dtype=NODE_DTYPE,
        ),
        edges=np.array([(0, 1, 1)], dtype=EDGE_DTYPE),
        record_offsets=np.array([0, 1, 2], dtype=np.uint32),
        record_ids=np.array(['a', 'b']),
        subgraphs=[[0, 1],],
    )
    assemblies = type('AssembliesStub', (), {
        'is_targets': np.array([True, False], dtype=np.bool_),
    })()

    args = next(markers._get_create_ck_args(graph, assemblies, 7, 10))
    groups_by_hash = dict(zip(args[0], args[1]))

    np.testing.assert_array_equal(groups_by_hash[np.uint64(10)], kmers[0:2])
    np.testing.assert_array_equal(groups_by_hash[np.uint64(20)], kmers[3:5])
