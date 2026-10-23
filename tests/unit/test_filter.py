import pickle

import numpy as np
import pytest

from seqwin.core import (
    EDGE_DTYPE, KMER_DTYPE, FilteredGraph, Signature
)
from seqwin.core._native import _filter_native

from _filter_helpers import _filter, _inputs, _paths


def _filter_distinct_weights(tmp_path, edge_weight_th):
    kmers = np.array(
        [(0, record) for _ in range(4) for record in (0, 1, 2, 3)],
        dtype=KMER_DTYPE,
    )
    offsets = np.arange(0, len(kmers) + 1, 4, dtype=np.uintp)
    prevalence = np.full(4, 4, dtype=np.uint32)
    edges = np.array(
        [(0, 1, 5), (1, 2, 3), (2, 3, 2)],
        dtype=EDGE_DTYPE,
    )
    record_offsets = np.array([0, 1, 2, 3, 4, 4], dtype=np.uint32)
    assembly_nodes = np.tile(np.arange(4, dtype=np.uintp), 4)
    node_offsets = np.array([0, 4, 8, 12, 16, 16], dtype=np.uintp)
    targets = np.array([True, True, True, True, False], dtype=np.bool_)
    edge_w_th_mul = np.nextafter(edge_weight_th / 2.8, np.inf)
    return _filter_native(
        kmers, offsets, prevalence, edges, record_offsets, assembly_nodes, node_offsets,
        _paths(tmp_path, 5), targets, None, 5, 10, .3, 5,
        0, None, .2, edge_w_th_mul, 1, None, 1.5, 1,
    )


def test_native_filter_preserves_ranges_and_original_indices(tmp_path):
    (filtered, signatures), scored = _filter(tmp_path)
    nodes, edges, subgraphs = filtered.nodes, filtered.edges, filtered.subgraphs

    assert isinstance(filtered, FilteredGraph)
    assert isinstance(signatures, list)
    assert all(isinstance(signature, Signature) for signature in signatures)

    np.testing.assert_array_equal(scored, [2, 2, 4, 1])
    assert filtered.total_tar == 2 and filtered.total_neg == 2
    assert filtered.e_absence_tar is None
    assert filtered.e_presence_neg is None
    assert filtered.penalty_th == .3
    assert filtered.edge_weight_th == pytest.approx(.42)
    assert filtered.min_nodes == 1 and filtered.max_nodes is None
    np.testing.assert_array_equal(nodes['idx'], [0, 1, 2, 3])
    np.testing.assert_array_equal(nodes['n_tar'], [2, 2, 2, 1])
    np.testing.assert_array_equal(nodes['n_neg'], [0, 0, 2, 0])
    np.testing.assert_allclose(nodes['penalty'], [0, 0, 1, .5])
    assert edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
    assert np.all(edges['first'] < len(scored))
    assert np.all(edges['second'] < len(scored))
    assert set(edges['first']) <= set(nodes['idx'])
    assert set(edges['second']) <= set(nodes['idx'])
    assert subgraphs == [[0, 1]]
    assert len(signatures) == 1
    signature = signatures[0]
    assert signature.subgraph_idx == 0
    assert signature.location.stop - signature.location.start == 5
    assert signature.n_tar == 2
    assert all(node_i < len(scored) for subgraph in subgraphs for node_i in subgraph)
    assert set(nodes['idx']) - set(subgraphs[0]) == {2, 3}


def test_automatic_threshold_from_minimizers_and_parallel_equivalence(tmp_path):
    first, _ = _filter(tmp_path, penalty_th=None, n_cpu=1)
    parallel, _ = _filter(tmp_path, penalty_th=None, n_cpu=4)
    expected = .5 * np.sqrt((1 / 14) * (2 / 7))
    assert first[0].penalty_th == pytest.approx(expected)
    assert first[0].e_absence_tar == pytest.approx(1 / 14)
    assert first[0].e_presence_neg == pytest.approx(2 / 7)
    assert parallel[0].e_absence_tar == pytest.approx(first[0].e_absence_tar)
    assert parallel[0].e_presence_neg == pytest.approx(first[0].e_presence_neg)
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
        signature.location.n_repeats,
        signature.n_tar,
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
    tmp_path, targets, expected_n_tar, expected_n_neg,
):
    result, _ = _filter(
        tmp_path,
        targets=np.array(targets, dtype=np.bool_), penalty_th=.5, n_cpu=4,
    )
    np.testing.assert_array_equal(result[0].nodes['n_tar'], expected_n_tar)
    np.testing.assert_array_equal(result[0].nodes['n_neg'], expected_n_neg)


def test_automatic_threshold_from_jaccard_and_cap(tmp_path):
    jaccard = np.full((4, 4), .5, dtype=np.float64)
    result, _ = _filter(tmp_path, penalty_th=None, jaccard=jaccard, penalty_th_cap=1)
    assert result[0].penalty_th == pytest.approx(.5 * np.sqrt((1 / 3) * (2 / 3)))
    assert result[0].e_absence_tar == pytest.approx(1 / 3)
    assert result[0].e_presence_neg == pytest.approx(2 / 3)
    capped, _ = _filter(tmp_path, penalty_th=None, jaccard=jaccard, penalty_th_cap=.1)
    assert capped[0].penalty_th == .1


def test_low_weight_edges_isolated_nodes_and_no_subgraph_error(tmp_path):
    with pytest.raises(RuntimeError, match='adjust|Try decrease'):
        _filter(tmp_path, edge_w_th_mul=1, min_nodes_floor=2)


@pytest.mark.parametrize(
    ('edge_weight_th', 'expected'),
    (
        (.5, [(0, 1, 5), (1, 2, 3), (2, 3, 2)]),
        (2.5, [(0, 1, 5), (1, 2, 3)]),
        (3, [(0, 1, 5)]),
    ),
)
def test_edge_pruning_retains_descending_prefix_with_strict_threshold(
    tmp_path, edge_weight_th, expected,
):
    result = _filter_distinct_weights(tmp_path, edge_weight_th)
    assert result[0].edges.tolist() == expected


def test_edge_pruning_rejects_all_edges(tmp_path):
    with pytest.raises(RuntimeError, match='adjust|Try decrease'):
        _filter_distinct_weights(tmp_path, 5)


def test_edge_pruning_handles_empty_edge_array(tmp_path):
    kmers, offsets, prevalence, _, record_offsets, targets = _inputs()
    edges = np.empty(0, dtype=EDGE_DTYPE)
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)
    with pytest.raises(RuntimeError, match='adjust|Try decrease'):
        _filter_native(
            kmers, offsets, prevalence, edges, record_offsets, assembly_nodes, node_offsets,
            _paths(tmp_path), targets,
            None, 5, 10, .3, 5, 0, None, .2, .3, 1, None, 1.5, 1,
        )


def _filter_target_support(tmp_path, support, edge_data, total_tar, n_cpu):
    # Low-support nodes also occur in every non-target assembly.
    occurrences = [
        tuple(range(count)) + tuple(range(total_tar, 5))
        if count <= 1 else tuple(range(count))
        for count in support
    ]
    kmers = []
    offsets = []
    prevalence = []
    for i, records in enumerate(occurrences):
        offsets.append(len(kmers))
        prevalence.append(len(records))
        kmers.extend((0, record) for record in records)
    assembly_nodes = []
    node_offsets = [0]
    for record in range(5):
        assembly_nodes.extend(
            i for i, records in enumerate(occurrences) if record in records
        )
        node_offsets.append(len(assembly_nodes))
    targets = np.arange(5) < total_tar
    # The integer pruning threshold is 1; weight-1 edges must be rejected.
    edge_w_th_mul = 1.5 / ((1.0 - .9) * total_tar)
    return _filter_native(
        np.array(kmers, dtype=KMER_DTYPE),
        np.array([*offsets, len(kmers)], dtype=np.uintp),
        np.array(prevalence, dtype=np.uint32),
        np.array(edge_data, dtype=EDGE_DTYPE),
        np.arange(6, dtype=np.uint32),
        np.array(assembly_nodes, dtype=np.uintp),
        np.array(node_offsets, dtype=np.uintp),
        _paths(tmp_path, 5), targets, None, 5, 10, .9, 5,
        0, None, .2, edge_w_th_mul, 1, None, 1.5, n_cpu,
    )


@pytest.mark.parametrize(('total_tar', 'n_cpu'), ((2, 1), (3, 4)))
def test_edge_pruning_requires_target_support_and_preserves_order(tmp_path, total_tar, n_cpu):
    filtered, _ = _filter_target_support(
        tmp_path,
        [1, 1, 2, 1, 1, 2, 2, 2],
        [(0, 1, 5), (2, 3, 4), (0, 1, 3), (4, 5, 3), (6, 7, 2), (2, 6, 1)],
        total_tar, n_cpu,
    )

    assert filtered.edge_weight_th == pytest.approx(1.5)
    # Neither endpoint qualifies at equality; first-only, second-only and both do.
    np.testing.assert_array_equal(filtered.nodes['idx'], [2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(filtered.nodes['n_tar'], [2, 1, 1, 2, 2, 2])
    assert filtered.edges.tolist() == [(2, 3, 4), (4, 5, 3), (6, 7, 2)]
    assert filtered.subgraphs == [[2, 3], [5, 4], [6, 7]]


@pytest.mark.parametrize('edge', ((0, 2, 4), (2, 0, 4)))
def test_edge_pruning_validates_candidates_before_target_count_access(tmp_path, edge):
    with pytest.raises(ValueError, match='Edge endpoint does not correspond to a node'):
        _filter_target_support(tmp_path, [1, 1], [edge], 2, 1)


def test_original_indices_are_preserved_after_isolated_node_removal(tmp_path):
    kmers, offsets, prevalence, _, record_offsets, targets = _inputs()
    edges = np.array([(1, 2, 1)], dtype=EDGE_DTYPE)
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)
    result = _filter_native(
        kmers, offsets, prevalence, edges, record_offsets, assembly_nodes, node_offsets,
        _paths(tmp_path), targets, None, 5, 10, 1.0, 5, 0, None, .2, .3, 1, None, 1.5, 1,
    )

    filtered, signatures = result
    filtered_nodes, filtered_edges, subgraphs = filtered.nodes, filtered.edges, filtered.subgraphs
    np.testing.assert_array_equal(filtered_nodes['idx'], [1, 2])
    np.testing.assert_array_equal(filtered_edges, edges)
    assert filtered_edges.tolist() == [(1, 2, 1)]
    assert subgraphs == [[1, 2]]
    assert len(signatures) == 1
    assert signatures[0].location.stop - signatures[0].location.start == 5
    assert signatures[0].location.n_kmers == 2


def test_jaccard_shape_validation(tmp_path):
    with pytest.raises(ValueError, match='Jaccard matrix shape'):
        _filter(tmp_path, penalty_th=None, jaccard=np.ones((2, 2), dtype=np.float64))


def test_filtered_arrays_keep_native_owner_alive(tmp_path):
    (filtered, _), _ = _filter(tmp_path)
    nodes = filtered.nodes
    edges = filtered.edges
    assert nodes.base is filtered
    assert edges.base is filtered
    del filtered
    np.testing.assert_array_equal(nodes['idx'], [0, 1, 2, 3])
    assert edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]


def test_native_results_pickle_round_trip(tmp_path):
    (filtered, signatures), _ = _filter(tmp_path)

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
    assert restored_signature.n_tar == signature.n_tar
    for name in ('assembly_idx', 'record_idx', 'start', 'stop', 'n_kmers', 'n_repeats'):
        assert getattr(restored_signature.location, name) == getattr(signature.location, name)

    restored_location = pickle.loads(pickle.dumps(signature.location, protocol=5))
    for name in ('assembly_idx', 'record_idx', 'start', 'stop', 'n_kmers', 'n_repeats'):
        assert getattr(restored_location, name) == getattr(signature.location, name)


def test_target_counts_include_non_target_only_nodes_and_exclude_isolated_nodes(tmp_path):
    kmers, offsets, prevalence, _, record_offsets, targets = _inputs()
    extra_kmers = np.array([(0, 2), (1, 3)], dtype=KMER_DTYPE)
    edges = np.array([(0, 1, 1), (1, 2, 1), (2, 4, 1)], dtype=EDGE_DTYPE)
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 4, 2, 4], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 9, 11], dtype=np.uintp)
    kmers = np.concatenate((kmers, extra_kmers))
    offsets = np.append(offsets, np.uintp(len(kmers)))
    prevalence = np.append(prevalence, np.uint32(2))
    before = prevalence.copy()
    prevalence.setflags(write=False)
    filtered, _ = _filter_native(
        kmers,
        offsets,
        prevalence,
        edges,
        record_offsets,
        assembly_nodes,
        node_offsets,
        _paths(tmp_path), targets, None, 5, 10, 2.0, 5, 0, None, .2, 0, 1, None, 1.5, 1,
    )

    np.testing.assert_array_equal(prevalence, before)
    np.testing.assert_array_equal(filtered.nodes['idx'], [0, 1, 2, 4])
    np.testing.assert_array_equal(filtered.nodes['n_tar'], [2, 2, 2, 0])
    np.testing.assert_array_equal(filtered.nodes['n_neg'], [0, 0, 2, 2])
    np.testing.assert_allclose(filtered.nodes['penalty'], [0, 0, 1, np.sqrt(2)])
    assert 3 not in filtered.nodes['idx']
    assert filtered.edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 4, 1)]
