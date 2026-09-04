import numpy as np
import pytest

from seqwin.config import EDGE_W, NODE_P
from seqwin.graph import EDGE_DTYPE, KMER_DTYPE, NODE_DTYPE, _filter_native
from seqwin.kmers import FilteredGraph


def _inputs():
    occurrences = {10: (0, 1), 20: (0, 1), 30: (0, 1, 2, 3), 40: (0,)}
    kmers = []
    nodes = []
    for node_hash, records in occurrences.items():
        start = len(kmers)
        kmers.extend((i, record) for i, record in enumerate(records))
        nodes.append((node_hash, start, len(kmers), 0, 0, 0.0))
    return (
        np.array(kmers, dtype=KMER_DTYPE),
        np.array(nodes, dtype=NODE_DTYPE),
        np.array([(10, 20, 1), (20, 30, 1), (30, 40, 1)], dtype=EDGE_DTYPE),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
        np.array([True, True, False, False], dtype=np.bool_),
    )


def _filter(*, penalty_th=0.3, jaccard=None, n_cpu=1,
            penalty_th_cap=0.2, edge_w_th_mul=0.3, min_nodes_floor=1,
            max_nodes_cap=None):
    kmers, nodes, edges, offsets, targets = _inputs()
    result = _filter_native(
        kmers, nodes, edges, offsets, targets, jaccard, penalty_th, 5,
        penalty_th_cap, edge_w_th_mul, 10, 0, None, min_nodes_floor,
        max_nodes_cap, n_cpu,
    )
    return result, nodes


def test_native_filter_scores_compacts_and_filters_final_edges():
    result, scored = _filter()
    kmers, nodes, edges, subgraphs, penalty_th, edge_th, min_nodes, max_nodes = result

    np.testing.assert_array_equal(scored['n_tar'], [2, 2, 2, 1])
    np.testing.assert_array_equal(scored['n_neg'], [0, 0, 2, 0])
    np.testing.assert_allclose(scored['penalty'], [0, 0, 1, .5])
    assert penalty_th == .3
    assert edge_th == pytest.approx(.42)
    assert min_nodes == 1 and max_nodes is None
    np.testing.assert_array_equal(nodes['hash'], [10, 20])
    np.testing.assert_array_equal(nodes[['start', 'stop']].tolist(), [(0, 2), (2, 4)])
    assert len(kmers) == 4
    assert edges.tolist() == [(10, 20, 1)]
    assert set(edges['first']) | set(edges['second']) <= set(nodes['hash'])
    assert {frozenset(s) for s in subgraphs} == {frozenset((10, 20))}


def test_automatic_threshold_from_minimizers_and_parallel_equivalence():
    first, _ = _filter(penalty_th=None, n_cpu=1)
    parallel, _ = _filter(penalty_th=None, n_cpu=4)
    expected = .5 * np.sqrt((1 / 14) * (2 / 7))
    assert first[4] == pytest.approx(expected)
    for left, right in zip(first[:4], parallel[:4]):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_automatic_threshold_from_jaccard_and_cap():
    jaccard = np.full((4, 4), .5, dtype=np.float64)
    result, _ = _filter(penalty_th=None, jaccard=jaccard, penalty_th_cap=1)
    assert result[4] == pytest.approx(.5 * np.sqrt((1 / 3) * (2 / 3)))
    capped, _ = _filter(penalty_th=None, jaccard=jaccard, penalty_th_cap=.1)
    assert capped[4] == .1


def test_low_weight_edges_isolated_nodes_and_no_subgraph_error():
    with pytest.raises(RuntimeError, match='adjust|Try decrease'):
        _filter(edge_w_th_mul=1, min_nodes_floor=2)


def test_subgraph_extraction_is_deterministic():
    first, _ = _filter()
    second, _ = _filter()
    assert first[3] == second[3]


def test_jaccard_shape_validation():
    with pytest.raises(ValueError, match='Jaccard matrix shape'):
        _filter(penalty_th=None, jaccard=np.ones((2, 2), dtype=np.float64))


def test_filtered_graph_builds_exact_networkx_graph_and_attributes():
    result, _ = _filter()
    kmers, nodes, edges, subgraphs = result[:4]
    graph = FilteredGraph(kmers, nodes, edges, np.array([0, 1], dtype=np.uint32),
                          np.array(['record']), tuple(frozenset(s) for s in subgraphs))
    assert set(graph.nx_graph) == set(nodes['hash'])
    assert set(graph.nx_graph.edges) == {(np.uint64(10), np.uint64(20))}
    assert graph.nx_graph[10][20][EDGE_W] == 1
    assert graph.nx_graph.nodes[10][NODE_P] == 0
    assert set(graph.nx_graph.subgraph(graph.subgraphs[0])) == set(graph.subgraphs[0])
