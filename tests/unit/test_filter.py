from heapq import heappop, heappush
from random import Random

import networkx as nx
import numpy as np
import pytest

from seqwin.graph import EDGE_DTYPE, NODE_DTYPE, _get_subgraphs_native
from seqwin.kmers import _get_subgraphs


def _arrays(penalties, edge_rows):
    nodes = np.array([
        (node_hash, 0, 0, 0, 0, penalty)
        for node_hash, penalty in sorted(penalties.items())
    ], dtype=NODE_DTYPE)
    edges = np.array([
        (first, second, 1) for first, second in edge_rows
    ], dtype=EDGE_DTYPE)
    return nodes, edges


def _reference(nodes, edges, penalty_th, min_nodes, max_nodes, rng):
    graph = nx.Graph()
    graph.add_edges_from((edge['first'], edge['second']) for edge in edges)
    nx.set_node_attributes(
        graph,
        dict(zip(nodes['hash'], nodes['penalty'])),
        'penalty',
    )
    node_penalty = dict(sorted(graph.nodes(data='penalty')))
    seeds = [node for node, penalty in node_penalty.items() if penalty <= penalty_th]
    rng.shuffle(seeds)

    used = set()
    subgraphs = []
    for seed in seeds:
        if seed in used:
            continue
        subgraph = {seed}
        sum_penalty = node_penalty[seed]
        frontier_heap = []
        frontier_set = set()
        for neighbor in graph.neighbors(seed):
            if neighbor not in used and neighbor not in subgraph:
                heappush(frontier_heap, (node_penalty[neighbor], neighbor))
                frontier_set.add(neighbor)
        while frontier_heap and (max_nodes is None or len(subgraph) < max_nodes):
            penalty, node = heappop(frontier_heap)
            if node not in frontier_set:
                continue
            new_sum_penalty = sum_penalty + penalty
            if new_sum_penalty / (len(subgraph) + 1) <= penalty_th:
                subgraph.add(node)
                sum_penalty = new_sum_penalty
                for neighbor in graph.neighbors(node):
                    if neighbor not in used and neighbor not in subgraph and neighbor not in frontier_set:
                        heappush(frontier_heap, (node_penalty[neighbor], neighbor))
                        frontier_set.add(neighbor)
            frontier_set.remove(node)
        if len(subgraph) >= min_nodes:
            subgraphs.append(subgraph)
            used |= subgraph

    if not subgraphs:
        raise RuntimeError('No low-penalty subgraph was found')
    rng.shuffle(subgraphs)
    return tuple(frozenset(sg) for sg in subgraphs), frozenset(used)


CASES = (
    # Equal penalties exercise the hash tie-break; competing seeds become used.
    ({10: 0.1, 20: 0.2, 30: 0.2, 40: 0.1}, [(10, 30), (10, 20), (20, 40)], 0.2, 2, None),
    # A high-penalty frontier node is rejected while disconnected components remain.
    ({1: 0.0, 2: 0.9, 3: 0.0, 4: 0.0}, [(1, 2), (3, 4)], 0.1, 1, None),
    # Small candidates are discarded and their nodes can be reused by later seeds.
    ({1: 0.1, 2: 0.3, 3: 0.1}, [(1, 2), (2, 3)], 0.2, 3, None),
    # Expansion stops exactly at max_nodes.
    ({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, [(1, 2), (2, 3), (3, 4)], 0.0, 2, 2),
    # Zero threshold, a self-loop, and reversed edge-row order.
    ({5: 0.0, 6: 0.0, 7: 0.0}, [(6, 7), (5, 6), (5, 5)], 0.0, 1, None),
)


@pytest.mark.parametrize(('penalties', 'edge_rows', 'penalty_th', 'min_nodes', 'max_nodes'), CASES)
@pytest.mark.parametrize('seed', (0, 7, 991))
def test_native_subgraphs_match_python_reference(
    penalties, edge_rows, penalty_th, min_nodes, max_nodes, seed
):
    nodes, edges = _arrays(penalties, edge_rows)
    rng_ref = Random(seed)
    rng_native = Random(seed)

    expected = _reference(nodes, edges, penalty_th, min_nodes, max_nodes, rng_ref)
    actual = _get_subgraphs(nodes, edges, penalty_th, min_nodes, max_nodes, rng_native)

    assert actual == expected
    assert rng_native.getstate() == rng_ref.getstate()
    assert all(isinstance(node, np.uint64) for subgraph in actual[0] for node in subgraph)


def test_native_creation_order_is_independent_of_edge_row_order():
    penalties = {1: 0.0, 2: 0.1, 3: 0.1, 4: 0.0}
    nodes, edges = _arrays(penalties, [(1, 3), (1, 2), (2, 4)])
    seeds = [np.uint64(1), np.uint64(4)]
    first = _get_subgraphs_native(nodes, edges, seeds, 0.1, 1, None)
    second = _get_subgraphs_native(nodes, edges[::-1].copy(), seeds, 0.1, 1, None)
    assert first == second


def test_no_valid_subgraph_preserves_error_and_rng_state():
    nodes, edges = _arrays({1: 0.0, 2: 0.0}, [(1, 2)])
    rng_ref = Random(42)
    rng_native = Random(42)

    with pytest.raises(RuntimeError, match='No low-penalty subgraph was found'):
        _reference(nodes, edges, 0.0, 3, None, rng_ref)
    with pytest.raises(RuntimeError, match='No low-penalty subgraph was found'):
        _get_subgraphs(nodes, edges, 0.0, 3, None, rng_native)

    assert rng_native.getstate() == rng_ref.getstate()


def test_filtered_graph_preserves_ordered_subgraphs():
    from seqwin.graph import KMER_DTYPE, KmerGraph
    from seqwin.kmers import _get_filtered_graph

    graph = KmerGraph.__new__(KmerGraph)
    graph.kmers = np.array([(10, 0), (20, 0), (30, 0), (40, 0)], dtype=KMER_DTYPE)
    graph.nodes, graph.edges = _arrays(
        {10: 0.0, 20: 0.0, 30: 0.0, 40: 0.0},
        [(10, 20), (30, 40)],
    )
    graph.nodes['start'] = np.arange(4)
    graph.nodes['stop'] = np.arange(1, 5)
    graph.record_offsets = np.array([0, 1], dtype=np.uint32)
    graph.record_ids = np.array(['record'])

    filtered = _get_filtered_graph(graph, 0.0, 0.0, 2, None, Random(11))

    assert filtered.subgraphs == (
        frozenset((np.uint64(10), np.uint64(20))),
        frozenset((np.uint64(30), np.uint64(40))),
    )
    assert set(filtered.nx_graph.edges) == {(np.uint64(10), np.uint64(20)),
                                           (np.uint64(30), np.uint64(40))}
