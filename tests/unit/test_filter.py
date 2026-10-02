import pickle

import numpy as np
import pytest

from seqwin.core import (
    EDGE_DTYPE, KMER_DTYPE, NODE_DTYPE, FilteredGraph, Signature
)
from seqwin.core._native import _filter_native


def _paths():
    paths = []
    for i in range(4):
        path = f'/tmp/seqwin-filter-{i}.fasta'
        with open(path, 'w') as fasta:
            fasta.write(f'>record-{i}\nACGTACGTACGT\n')
        paths.append(path)
    return paths


def _inputs():
    occurrences = {10: (0, 1), 20: (0, 1), 30: (0, 1, 2, 3), 40: (0,)}
    kmers = []
    nodes = []
    for node_hash, records in occurrences.items():
        start = len(kmers)
        kmers.extend((i, record) for i, record in enumerate(records))
        nodes.append((node_hash, start, len(kmers), len(set(records))))
    return (
        np.array(kmers, dtype=KMER_DTYPE),
        np.array(nodes, dtype=NODE_DTYPE),
        np.array([(0, 1, 1), (1, 2, 1), (2, 3, 1)], dtype=EDGE_DTYPE),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
        np.array([True, True, False, False], dtype=np.bool_),
    )


def _filter(*, penalty_th=0.3, jaccard=None, n_cpu=1, targets=None,
            penalty_th_cap=0.2, edge_w_th_mul=0.3, min_nodes_floor=1,
            max_nodes_cap=None):
    kmers, nodes, edges, offsets, default_targets = _inputs()
    if targets is None:
        targets = default_targets
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)
    result = _filter_native(
        kmers, nodes, edges, offsets, assembly_nodes, node_offsets,
        _paths(), targets, jaccard, 5, 10,
        penalty_th, 5, 0, None, penalty_th_cap, edge_w_th_mul, min_nodes_floor,
        max_nodes_cap, 1.5, n_cpu,
    )
    return result, nodes


def _filter_distinct_weights(edge_weight_th):
    kmers = np.array(
        [(0, record) for _ in range(4) for record in (0, 1)],
        dtype=KMER_DTYPE,
    )
    nodes = np.array(
        [(node_hash, i * 2, i * 2 + 2, 2)
         for i, node_hash in enumerate((10, 20, 30, 40))],
        dtype=NODE_DTYPE,
    )
    edges = np.array(
        [(0, 1, 5), (1, 2, 3), (2, 3, 2)],
        dtype=EDGE_DTYPE,
    )
    record_offsets = np.array([0, 1, 2, 2, 2], dtype=np.uint32)
    assembly_nodes = np.tile(np.arange(4, dtype=np.uintp), 2)
    node_offsets = np.array([0, 4, 8, 8, 8], dtype=np.uintp)
    targets = np.array([True, True, False, False], dtype=np.bool_)
    edge_w_th_mul = np.nextafter(edge_weight_th / 1.4, np.inf)
    return _filter_native(
        kmers, nodes, edges, record_offsets, assembly_nodes, node_offsets,
        _paths(), targets, None, 5, 10, .3, 5,
        0, None, .2, edge_w_th_mul, 1, None, 1.5, 1,
    )


def test_native_filter_preserves_ranges_and_remaps_edges():
    (filtered, signatures), scored = _filter()
    nodes, edges, subgraphs = filtered.nodes, filtered.edges, filtered.subgraphs

    assert isinstance(filtered, FilteredGraph)
    assert isinstance(signatures, list)
    assert all(isinstance(signature, Signature) for signature in signatures)

    np.testing.assert_array_equal(scored['prevalence'], [2, 2, 4, 1])
    assert filtered.total_tar == 2 and filtered.total_neg == 2
    assert filtered.e_absence_tar == pytest.approx(1 / 14)
    assert filtered.e_presence_neg == pytest.approx(2 / 7)
    assert filtered.penalty_th == .3
    assert filtered.edge_weight_th == pytest.approx(.42)
    assert filtered.min_nodes == 1 and filtered.max_nodes is None
    np.testing.assert_array_equal(nodes['idx'], [0, 1, 2, 3])
    np.testing.assert_array_equal(nodes['n_tar'], [2, 2, 2, 1])
    np.testing.assert_array_equal(nodes['n_neg'], [0, 0, 2, 0])
    np.testing.assert_allclose(nodes['penalty'], [0, 0, 1, .5])
    assert edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
    assert np.all(edges['first'] < len(nodes))
    assert np.all(edges['second'] < len(nodes))
    assert nodes[edges['first']]['idx'].tolist() == [0, 1, 2]
    assert nodes[edges['second']]['idx'].tolist() == [1, 2, 3]
    assert subgraphs == [[0, 1]]
    assert len(signatures) == 1
    signature = signatures[0]
    assert signature.subgraph_idx == 0
    assert signature.sequence == 'ACGTA'
    assert signature.length == 5
    assert signature.n_rep == 2
    assert signature.rep_ratio == 1.0
    assert all(node_i < len(nodes) for subgraph in subgraphs for node_i in subgraph)
    np.testing.assert_array_equal(nodes[subgraphs[0]]['idx'], [0, 1])
    assert set(nodes['idx']) - set(nodes[subgraphs[0]]['idx']) == {2, 3}


def test_automatic_threshold_from_minimizers_and_parallel_equivalence():
    first, _ = _filter(penalty_th=None, n_cpu=1)
    parallel, _ = _filter(penalty_th=None, n_cpu=4)
    expected = .5 * np.sqrt((1 / 14) * (2 / 7))
    assert first[0].penalty_th == pytest.approx(expected)
    for left, right in ((first[0].nodes, parallel[0].nodes),
                        (first[0].edges, parallel[0].edges),
                        (first[0].subgraphs, parallel[0].subgraphs)):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    signature_fields = lambda signature: (
        signature.subgraph_idx, signature.location.assembly_idx,
        signature.location.record_idx, signature.location.start,
        signature.location.stop, signature.location.n_kmers,
        signature.location.n_repeats, signature.sequence,
        signature.length, signature.n_rep, signature.rep_ratio,
    )
    assert list(map(signature_fields, first[1])) == list(map(signature_fields, parallel[1]))


@pytest.mark.parametrize(
    ('targets', 'expected_n_tar', 'expected_n_neg'),
    (
        ([True, False, False, False], [1, 1, 1, 1], [1, 1, 3, 0]),
        ([True, True, True, False], [2, 2, 3, 1], [0, 0, 1, 0]),
    ),
)
def test_dense_counts_use_smaller_assembly_group_in_parallel(
    targets, expected_n_tar, expected_n_neg,
):
    result, _ = _filter(
        targets=np.array(targets, dtype=np.bool_), penalty_th=.5, n_cpu=4,
    )
    np.testing.assert_array_equal(result[0].nodes['n_tar'], expected_n_tar)
    np.testing.assert_array_equal(result[0].nodes['n_neg'], expected_n_neg)


def test_automatic_threshold_from_jaccard_and_cap():
    jaccard = np.full((4, 4), .5, dtype=np.float64)
    result, _ = _filter(penalty_th=None, jaccard=jaccard, penalty_th_cap=1)
    assert result[0].penalty_th == pytest.approx(.5 * np.sqrt((1 / 3) * (2 / 3)))
    capped, _ = _filter(penalty_th=None, jaccard=jaccard, penalty_th_cap=.1)
    assert capped[0].penalty_th == .1


def test_low_weight_edges_isolated_nodes_and_no_subgraph_error():
    with pytest.raises(RuntimeError, match='adjust|Try decrease'):
        _filter(edge_w_th_mul=1, min_nodes_floor=2)


@pytest.mark.parametrize(
    ('edge_weight_th', 'expected'),
    (
        (.5, [(0, 1, 5), (1, 2, 3), (2, 3, 2)]),
        (2.5, [(0, 1, 5), (1, 2, 3)]),
        (3, [(0, 1, 5)]),
    ),
)
def test_edge_pruning_retains_descending_prefix_with_strict_threshold(
    edge_weight_th, expected,
):
    result = _filter_distinct_weights(edge_weight_th)
    assert result[0].edges.tolist() == expected


@pytest.mark.parametrize('edge_weight_th', (5, 100))
def test_edge_pruning_rejects_all_edges(edge_weight_th):
    with pytest.raises(RuntimeError, match='adjust|Try decrease'):
        _filter_distinct_weights(edge_weight_th)


def test_edge_pruning_handles_empty_edge_array():
    kmers, nodes, _, offsets, targets = _inputs()
    edges = np.empty(0, dtype=EDGE_DTYPE)
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)
    with pytest.raises(RuntimeError, match='adjust|Try decrease'):
        _filter_native(
            kmers, nodes, edges, offsets, assembly_nodes, node_offsets,
            _paths(), targets,
            None, 5, 10, .3, 5, 0, None, .2, .3, 1, None, 1.5, 1,
        )


def test_edge_endpoints_are_remapped_after_isolated_node_removal():
    kmers, nodes, _, offsets, targets = _inputs()
    edges = np.array([(1, 2, 1)], dtype=EDGE_DTYPE)
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)
    result = _filter_native(
        kmers, nodes, edges, offsets, assembly_nodes, node_offsets,
        _paths(), targets, None, 5, 10, 1.0, 5, 0, None, .2, .3, 1, None, 1.5, 1,
    )

    filtered, _ = result
    filtered_nodes, filtered_edges, subgraphs = filtered.nodes, filtered.edges, filtered.subgraphs
    np.testing.assert_array_equal(filtered_nodes['idx'], [1, 2])
    assert filtered_edges.tolist() == [(0, 1, 1)]
    assert subgraphs == [[0, 1]]
    np.testing.assert_array_equal(filtered_nodes[subgraphs[0]]['idx'], [1, 2])


def test_subgraph_extraction_is_deterministic():
    first, _ = _filter()
    second, _ = _filter()
    assert first[0].subgraphs == second[0].subgraphs


def test_jaccard_shape_validation():
    with pytest.raises(ValueError, match='Jaccard matrix shape'):
        _filter(penalty_th=None, jaccard=np.ones((2, 2), dtype=np.float64))


def test_filtered_arrays_keep_native_owner_alive():
    (filtered, _), _ = _filter()
    nodes = filtered.nodes
    edges = filtered.edges
    assert nodes.base is filtered
    assert edges.base is filtered
    del filtered
    np.testing.assert_array_equal(nodes['idx'], [0, 1, 2, 3])
    assert edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]


def test_native_results_pickle_round_trip():
    (filtered, signatures), _ = _filter()

    restored = pickle.loads(pickle.dumps(filtered, protocol=5))
    np.testing.assert_array_equal(restored.nodes, filtered.nodes)
    np.testing.assert_array_equal(restored.edges, filtered.edges)
    for name in (
        'subgraphs', 'total_tar', 'total_neg', 'e_absence_tar',
        'e_presence_neg', 'penalty_th', 'edge_weight_th', 'min_nodes',
        'max_nodes',
    ):
        assert getattr(restored, name) == getattr(filtered, name)
    nodes, edges = restored.nodes, restored.edges
    assert nodes.base is restored
    assert edges.base is restored
    del restored
    np.testing.assert_array_equal(nodes['idx'], [0, 1, 2, 3])
    assert edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]

    signature = signatures[0]
    restored_signature = pickle.loads(pickle.dumps(signature, protocol=5))
    assert restored_signature.subgraph_idx == signature.subgraph_idx
    assert restored_signature.sequence == signature.sequence
    assert restored_signature.length == signature.length
    assert restored_signature.n_rep == signature.n_rep
    assert restored_signature.rep_ratio == signature.rep_ratio
    for name in ('assembly_idx', 'record_idx', 'start', 'stop', 'n_kmers', 'n_repeats'):
        assert getattr(restored_signature.location, name) == getattr(signature.location, name)

    restored_location = pickle.loads(pickle.dumps(signature.location, protocol=5))
    for name in ('assembly_idx', 'record_idx', 'start', 'stop', 'n_kmers', 'n_repeats'):
        assert getattr(restored_location, name) == getattr(signature.location, name)



def test_target_counts_include_non_target_only_nodes_and_exclude_isolated_nodes():
    kmers, original_nodes, _, offsets, targets = _inputs()
    extra_kmers = np.array([(0, 2), (1, 3)], dtype=KMER_DTYPE)
    extra_nodes = np.array([(50, 9, 11, 2)], dtype=NODE_DTYPE)
    edges = np.array([(0, 1, 1), (1, 2, 1), (2, 4, 1)], dtype=EDGE_DTYPE)
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 4, 2, 4], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 9, 11], dtype=np.uintp)
    kmers = np.concatenate((kmers, extra_kmers))
    original_nodes = np.concatenate((original_nodes, extra_nodes))
    before = original_nodes.copy()
    original_nodes.setflags(write=False)
    filtered, _ = _filter_native(
        kmers,
        original_nodes,
        edges,
        offsets,
        assembly_nodes,
        node_offsets,
        _paths(), targets, None, 5, 10, 2.0, 5, 0, None, .2, 0, 1, None, 1.5, 1,
    )

    np.testing.assert_array_equal(original_nodes, before)
    np.testing.assert_array_equal(filtered.nodes['idx'], [0, 1, 2, 4])
    np.testing.assert_array_equal(filtered.nodes['n_tar'], [2, 2, 2, 0])
    np.testing.assert_array_equal(filtered.nodes['n_neg'], [0, 0, 2, 2])
    np.testing.assert_allclose(filtered.nodes['penalty'], [0, 0, 1, np.sqrt(2)])
    assert 3 not in filtered.nodes['idx']
    assert filtered.edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
