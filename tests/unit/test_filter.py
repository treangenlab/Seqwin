import pickle
from pathlib import Path

import numpy as np
import pytest

from seqwin.core import AssemblyPaths, FilteredGraph, Signature
from seqwin.core._native import _filter_native

from _filter_helpers import EDGE_DTYPE, KMER_DTYPE, _arrays, _filter, _graph, _inputs, _paths


def _signature_values(signature):
    location = signature.location
    return (
        signature.subgraph_idx, signature.n_tar,
        location.assembly_idx, location.record_idx, location.start, location.stop,
        location.n_kmers, location.n_repeats,
    )


def _filter_distinct_weights(tmp_path, edge_weight_th):
    kmers = np.array(
        [(0, record) for _ in range(4) for record in (0, 1, 2, 3)],
        dtype=KMER_DTYPE,
    )
    offsets = np.arange(0, len(kmers) + 1, 4, dtype=np.uintp)
    prevalences = np.full(4, 4, dtype=np.uint32)
    edges = np.array(
        [(0, 1, 5), (1, 2, 3), (2, 3, 2)],
        dtype=EDGE_DTYPE,
    )
    record_offsets = np.array([0, 1, 2, 3, 4, 4], dtype=np.uint32)
    assembly_nodes = np.tile(np.arange(4, dtype=np.uintp), 4)
    node_offsets = np.array([0, 4, 8, 12, 16, 16], dtype=np.uintp)
    targets = np.array([True, True, True, True, False], dtype=np.bool_)
    edge_w_th_mul = edge_weight_th / 4
    graph = _graph(
        kmers, offsets, prevalences, edges, record_offsets, assembly_nodes, node_offsets,
    )
    return _filter_native(
        graph,
        _paths(tmp_path, 5), targets, 5, 10, .5, .2,
        0, None, edge_w_th_mul, None, None, 1, None, 1.5, 1,
    )


def test_native_filter_preserves_ranges_and_original_indices(tmp_path):
    (filtered, signatures), scored = _filter(tmp_path)
    nodes, edges, subgraphs = filtered.nodes, filtered.edges, filtered.subgraphs

    assert isinstance(filtered, FilteredGraph)
    assert isinstance(signatures, list)
    assert all(isinstance(signature, Signature) for signature in signatures)

    np.testing.assert_array_equal(scored, [2, 2, 4, 1])
    assert filtered.total_tar == 2 and filtered.total_neg == 2
    assert filtered.n_tar_th == 2
    assert filtered.n_neg_th == 0
    assert nodes.dtype == np.dtype([('idx', np.uintp), ('n_tar', np.uint32), ('n_neg', np.uint32)])
    assert filtered.edge_weight_th == pytest.approx(.6)
    assert filtered.min_nodes == 1 and filtered.max_nodes is None
    np.testing.assert_array_equal(nodes['idx'], [0, 1, 2, 3])
    np.testing.assert_array_equal(nodes['n_tar'], [2, 2, 2, 1])
    np.testing.assert_array_equal(nodes['n_neg'], [0, 0, 2, 0])
    assert edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
    assert subgraphs == [[0, 1]]
    assert len(signatures) == 1
    signature = signatures[0]
    assert signature.subgraph_idx == 0
    assert signature.location.stop - signature.location.start == 5
    assert signature.n_tar == 2


def test_automatic_thresholds_and_parallel_equivalence(tmp_path):
    first, _ = _filter(tmp_path, n_cpu=1)
    parallel, _ = _filter(tmp_path, n_cpu=4)
    assert first[0].n_tar_th == parallel[0].n_tar_th == 2
    assert first[0].n_neg_th == parallel[0].n_neg_th == 0
    for left, right in ((first[0].nodes, parallel[0].nodes),
                        (first[0].edges, parallel[0].edges),
                        (first[0].subgraphs, parallel[0].subgraphs)):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    assert list(map(_signature_values, first[1])) == list(map(_signature_values, parallel[1]))


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
        targets=np.array(targets, dtype=np.bool_), n_tar_th=1, n_neg_th=1, n_cpu=4,
    )
    np.testing.assert_array_equal(result[0].nodes['n_tar'], expected_n_tar)
    np.testing.assert_array_equal(result[0].nodes['n_neg'], expected_n_neg)


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


def _filter_counts(tmp_path, counts, edges, total_tar, total_neg, **options):
    occurrences = [
        (*range(n_tar), *range(total_tar, total_tar + n_neg))
        for n_tar, n_neg in counts
    ]
    n_assemblies = total_tar + total_neg
    kmers = []
    offsets = []
    prevalences = []
    for records in occurrences:
        offsets.append(len(kmers))
        prevalences.append(len(records))
        kmers.extend((0, record) for record in records)
    assembly_nodes = []
    node_offsets = [0]
    for record in range(n_assemblies):
        assembly_nodes.extend(
            i for i, records in enumerate(occurrences) if record in records
        )
        node_offsets.append(len(assembly_nodes))
    graph = _graph(
        np.array(kmers, dtype=KMER_DTYPE),
        np.array([*offsets, len(kmers)], dtype=np.uintp),
        np.array(prevalences, dtype=np.uint32),
        np.array(edges, dtype=EDGE_DTYPE),
        np.arange(n_assemblies + 1, dtype=np.uint32),
        np.array(assembly_nodes, dtype=np.uintp),
        np.array(node_offsets, dtype=np.uintp),
    )
    config = dict(
        kmerlen=5, windowsize=10, tar_weight=.5,
        neg_rank=.2, min_len=0, max_len=None, edge_w_th_mul=0, n_tar_th=None, n_neg_th=None,
        min_nodes_floor=1, max_nodes_cap=None, consec_kmer_mul=1.5,
    )
    config.update(options)
    filtered, _ = _filter_native(
        graph, _paths(tmp_path, n_assemblies), np.arange(n_assemblies) < total_tar, **config,
    )
    return filtered


@pytest.mark.parametrize('n_cpu', (1, 4))
@pytest.mark.parametrize(('tar_weight', 'threshold'), ((0, 4), (.2, 5), (.5, 5), (1, 6)))
def test_weighted_mean_uses_retained_nodes_only(tmp_path, tar_weight, threshold, n_cpu):
    # Each block has weighted mean 96 / 24 = 4. Including isolated node 0 would raise it.
    block = [(1, 0)] * 8 + [(4, 0), (6, 0), (6, 0)]
    counts = [(6, 0)] + block * 32
    edges = [(i, i + 1, 1) for i in range(1, len(counts) - 1)]
    filtered = _filter_counts(
        tmp_path, counts, edges, 6, 2, tar_weight=tar_weight, n_neg_th=2, n_cpu=n_cpu,
    )
    assert filtered.n_tar_th == threshold
    assert filtered.n_neg_th == 2


@pytest.mark.parametrize(('quantile', 'expected'), ((0, 0), (.4, 0), (.41, 1), (.8, 1), (1, 3)))
def test_nearest_rank_quantile_after_manual_target_override(tmp_path, quantile, expected):
    # Eligible non-target counts sort to [0, 0, 1, 1, 3]. Node 0 is ineligible.
    counts = [(2, 4), (4, 3), (3, 1), (4, 0), (3, 0), (4, 1)]
    edges = [(i, i + 1, 1) for i in range(5)]
    filtered = _filter_counts(
        tmp_path, counts, edges, 4, 4, n_tar_th=3, tar_weight=1, neg_rank=quantile,
    )
    assert filtered.n_tar_th == 3
    assert filtered.n_neg_th == expected


@pytest.mark.parametrize('options', ({'n_tar_th': 3}, {'n_neg_th': 3}))
def test_native_threshold_validation(tmp_path, options):
    with pytest.raises(ValueError, match=next(iter(options))):
        _filter(tmp_path, **options)


@pytest.mark.parametrize('name', ('tar_weight', 'neg_rank'))
@pytest.mark.parametrize('value', (-.01, 1.01, float('nan'), float('inf'), float('-inf')))
def test_native_prevalence_parameter_validation(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        _filter(tmp_path, **{name: value})


@pytest.mark.parametrize('options', ({'edge_w_th_mul': 1}, {'tar_weight': 1}))
def test_no_retained_or_target_eligible_nodes(tmp_path, options):
    with pytest.raises(RuntimeError, match='No subgraph'):
        _filter_counts(tmp_path, [(1, 0), (2, 1)], [(0, 1, 2)], 4, 2, **options)


def test_seeds_must_meet_both_thresholds(tmp_path):
    filtered = _filter_counts(
        tmp_path, [(3, 1), (3, 1), (2, 0), (2, 0), (4, 2), (4, 2), (4, 0), (4, 0)],
        [(0, 1, 1), (2, 3, 1), (4, 5, 1), (6, 7, 1), (1, 6, 1)],
        4, 2, n_tar_th=3, n_neg_th=1, max_nodes_cap=2,
    )
    assert filtered.subgraphs == [[0, 1], [6, 7]]


def test_frontier_uses_chebyshev_then_additive_score_then_node_index(tmp_path):
    # Nodes 3 and 4 tie at (.75, 1.25), followed by 2 at (.75, 1.5), then 1 at (1, 1).
    filtered = _filter_counts(
        tmp_path, [(6, 0), (2, 0), (3, 3), (4, 3), (4, 3)], [(0, i, 1) for i in (4, 3, 2, 1)],
        6, 4, n_tar_th=2, n_neg_th=4,
    )
    assert filtered.subgraphs == [[0, 3, 4, 2, 1]]


@pytest.mark.parametrize('counts', (
    [(4, 0), (3, 0), (6, 4)],  # Node 1 exceeds the target absence budget.
    [(6, 2), (6, 3), (2, 0)],  # Node 1 exceeds the non-target presence budget.
))
def test_expansion_continues_without_reconsidering_rejected_neighbors(tmp_path, counts):
    # Node 1 fails first. Node 2 replenishes the budget and also neighbors node 1.
    filtered = _filter_counts(
        tmp_path, counts, [(0, 1, 1), (0, 2, 1), (1, 2, 1)], 6, 4, n_tar_th=4, n_neg_th=2,
    )
    assert filtered.subgraphs == [[0, 2]]


def test_both_budgets_can_be_spent_to_accept_an_imperfect_node(tmp_path):
    filtered = _filter_counts(
        tmp_path, [(6, 0), (2, 4)], [(0, 1, 1)], 6, 4, n_tar_th=4, n_neg_th=2,
    )
    assert filtered.subgraphs == [[0, 1]]


def test_unsuccessful_expansion_releases_all_discovered_nodes(tmp_path):
    # Seed 0 cannot afford node 1. Seed 2 can cross node 1 to reach seed 0.
    filtered = _filter_counts(
        tmp_path, [(4, 2), (3, 0), (6, 2)], [(0, 1, 1), (1, 2, 1)],
        6, 2, n_tar_th=4, n_neg_th=2, min_nodes_floor=3,
    )
    assert filtered.subgraphs == [[2, 1, 0]]


def test_next_seed_can_accept_node_rejected_by_successful_expansion(tmp_path):
    # Seed 0 claims node 1 but cannot afford node 2. Seed 3 can afford it.
    filtered = _filter_counts(
        tmp_path, [(4, 0), (4, 0), (3, 0), (5, 0)], [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        5, 1, n_tar_th=4, n_neg_th=0, min_nodes_floor=2,
    )
    assert filtered.subgraphs == [[0, 1], [3, 2]]


@pytest.mark.parametrize(('n_tar_th', 'n_neg_th', 'expected'), (
    (4, 1, [[0, 2, 3]]), (3, 0, [[0, 2, 1]]), (4, 0, [[0, 2]]),
))
def test_zero_tolerance_frontier(tmp_path, n_tar_th, n_neg_th, expected):
    filtered = _filter_counts(
        tmp_path, [(4, 0), (3, 0), (4, 0), (4, 1)], [(0, i, 1) for i in (1, 2, 3)],
        4, 2, n_tar_th=n_tar_th, n_neg_th=n_neg_th,
    )
    assert filtered.subgraphs == expected


def _filter_target_support(tmp_path, support, edges, total_tar, n_cpu):
    # Low-support nodes also occur in every non-target assembly.
    total_neg = 5 - total_tar
    counts = [(count, total_neg if count <= 1 else 0) for count in support]
    return _filter_counts(
        tmp_path, counts, edges, total_tar, total_neg, n_cpu=n_cpu,
        n_tar_th=1, n_neg_th=(total_neg + 1) // 2, edge_w_th_mul=1.5 / total_tar,
    )


@pytest.mark.parametrize(('total_tar', 'n_cpu'), ((2, 1), (3, 4)))
def test_edge_pruning_requires_target_support_and_preserves_order(tmp_path, total_tar, n_cpu):
    filtered = _filter_target_support(
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
    kmers, offsets, prevalences, _, record_offsets, targets = _inputs()
    edges = np.array([(1, 2, 1)], dtype=EDGE_DTYPE)
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)
    graph = _graph(
        kmers, offsets, prevalences, edges, record_offsets, assembly_nodes, node_offsets,
    )
    result = _filter_native(
        graph,
        _paths(tmp_path), targets, 5, 10, .5, .2, 0, None, .3, 2, 2, 1, None, 1.5, 1,
    )

    filtered, signatures = result
    filtered_nodes, filtered_edges, subgraphs = filtered.nodes, filtered.edges, filtered.subgraphs
    np.testing.assert_array_equal(filtered_nodes['idx'], [1, 2])
    np.testing.assert_array_equal(filtered_edges, edges)
    assert subgraphs == [[1, 2]]
    assert len(signatures) == 1
    assert signatures[0].location.stop - signatures[0].location.start == 5
    assert signatures[0].location.n_kmers == 2


def test_filtered_arrays_keep_native_owner_alive(tmp_path):
    (filtered, _), _ = _filter(tmp_path)
    nodes = filtered.nodes
    edges = filtered.edges
    assert nodes.base is filtered
    assert edges.base is filtered
    for array, repeated in ((nodes, filtered.nodes), (edges, filtered.edges)):
        assert np.shares_memory(array, repeated)
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0] = array[0]
    del filtered
    np.testing.assert_array_equal(nodes['idx'], [0, 1, 2, 3])
    assert edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 3, 1)]


def test_native_results_pickle_round_trip(tmp_path):
    (filtered, signatures), _ = _filter(tmp_path)

    assert filtered.__getstate__()[0] == 1
    restored = pickle.loads(pickle.dumps(filtered, protocol=5))
    np.testing.assert_array_equal(restored.nodes, filtered.nodes)
    np.testing.assert_array_equal(restored.edges, filtered.edges)
    for name in (
        'subgraphs', 'total_tar', 'total_neg', 'n_tar_th', 'n_neg_th',
        'edge_weight_th', 'min_nodes', 'max_nodes',
    ):
        assert getattr(restored, name) == getattr(filtered, name)
    signature = signatures[0]
    restored_signature = pickle.loads(pickle.dumps(signature, protocol=5))
    assert _signature_values(restored_signature) == _signature_values(signature)

    restored_location = pickle.loads(pickle.dumps(signature.location, protocol=5))
    for name in ('assembly_idx', 'record_idx', 'start', 'stop', 'n_kmers', 'n_repeats'):
        assert getattr(restored_location, name) == getattr(signature.location, name)


def test_target_counts_include_non_target_only_nodes_and_exclude_isolated_nodes(tmp_path):
    kmers, offsets, prevalences, _, record_offsets, targets = _inputs()
    extra_kmers = np.array([(0, 2), (1, 3)], dtype=KMER_DTYPE)
    edges = np.array([(0, 1, 1), (1, 2, 1), (2, 4, 1)], dtype=EDGE_DTYPE)
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 4, 2, 4], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 9, 11], dtype=np.uintp)
    kmers = np.concatenate((kmers, extra_kmers))
    offsets = np.append(offsets, np.uintp(len(kmers)))
    prevalences = np.append(prevalences, np.uint32(2))
    graph = _graph(
        kmers, offsets, prevalences, edges, record_offsets, assembly_nodes, node_offsets,
    )
    filtered, _ = _filter_native(
        graph,
        _paths(tmp_path), targets, 5, 10, .5, .2, 0, None, 0, 0, 2, 1, None, 1.5, 1,
    )

    np.testing.assert_array_equal(filtered.nodes['idx'], [0, 1, 2, 4])
    np.testing.assert_array_equal(filtered.nodes['n_tar'], [2, 2, 2, 0])
    np.testing.assert_array_equal(filtered.nodes['n_neg'], [0, 0, 2, 2])
    assert filtered.edges.tolist() == [(0, 1, 1), (1, 2, 1), (2, 4, 1)]


@pytest.mark.parametrize('mapped', (False, True))
def test_built_and_numpy_filter_results_match(tmp_path, targets_dir, non_targets_dir, mapped):
    from seqwin.core import Graph

    paths = [str(path) for directory in (targets_dir, non_targets_dir)
             for path in sorted(directory.glob('*.fasta'))]
    graph = Graph(paths, 17, 10)
    if mapped:
        graph.save(tmp_path)
        loaded = Graph.load(tmp_path)
    else:
        arrays = {name: array.copy() for name, array in _arrays(graph).items()}
        for array in arrays.values():
            array.setflags(write=False)
        loaded = Graph._from_numpy(**arrays)
    targets = np.array([True, True, False, False], dtype=np.bool_)
    results = [
        _filter_native(
            graph=value, assembly_paths=paths, is_targets=targets,
            kmerlen=17, windowsize=10, tar_weight=.5, neg_rank=.2,
            min_len=17, max_len=200, edge_w_th_mul=.3, n_tar_th=None, n_neg_th=None,
            min_nodes_floor=1, max_nodes_cap=None, consec_kmer_mul=1.5, n_cpu=2,
        )
        for value in (graph, loaded)
    ]
    original, numpy_backed = results
    np.testing.assert_array_equal(original[0].nodes, numpy_backed[0].nodes)
    np.testing.assert_array_equal(original[0].edges, numpy_backed[0].edges)
    assert original[0].subgraphs == numpy_backed[0].subgraphs
    assert original[1]
    assert list(map(_signature_values, original[1])) == list(map(_signature_values, numpy_backed[1]))
    for name in ('total_tar', 'total_neg', 'n_tar_th', 'n_neg_th',
                 'edge_weight_th', 'min_nodes', 'max_nodes'):
        assert getattr(original[0], name) == getattr(numpy_backed[0], name)


class _CountingPath:
    def __init__(self, path):
        self.path = path
        self.calls = 0

    def __fspath__(self):
        self.calls += 1
        return self.path

    def __str__(self):
        raise AssertionError('Paths must use __fspath__, not __str__')


def _assembly_path_graph():
    kmers, offsets, prevalences, edges, record_offsets, _ = _inputs()
    return _graph(
        kmers, offsets, prevalences, edges, record_offsets,
        np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp),
        np.array([0, 4, 7, 8, 9], dtype=np.uintp),
    )


def _filter_assembly_paths(paths, targets=None, **output_paths):
    if targets is None:
        targets = _inputs()[-1]
    return _filter_native(
        _assembly_path_graph(), paths, targets,
        5, 10, .5, .2, 0, None, 0, 1, 2, 1, None, 1.5, 1, **output_paths,
    )


def test_filter_accepts_assembly_paths(tmp_path):
    paths = _paths(tmp_path)
    for i, path in enumerate(paths):
        Path(path).write_text(f'>record-{i}\n' + 'ACGT'[i] * 12 + '\n')
    paths.reverse()

    expected_fasta = tmp_path / 'expected.fasta'
    expected = _filter_assembly_paths(
        paths, fasta_path=str(expected_fasta), csv_path=str(tmp_path / 'expected.csv'),
    )

    mixed = tuple(Path(path) if i % 2 else path for i, path in enumerate(paths))
    for i, assembly_paths in enumerate((mixed, AssemblyPaths(mixed))):
        actual_fasta = tmp_path / f'actual-{i}.fasta'
        actual = _filter_assembly_paths(
            assembly_paths, fasta_path=str(actual_fasta), csv_path=str(tmp_path / f'actual-{i}.csv'),
        )

        np.testing.assert_array_equal(actual[0].nodes, expected[0].nodes)
        assert actual[0].subgraphs == expected[0].subgraphs
        assert list(map(_signature_values, actual[1])) == list(map(_signature_values, expected[1]))
        assert actual_fasta.read_text() == expected_fasta.read_text()


def test_assembly_paths_reused_without_normalizing_again(tmp_path):
    inputs = [_CountingPath(path) for path in _paths(tmp_path)]
    paths = AssemblyPaths(inputs)

    assert len(paths) == len(inputs)
    assert all(path.calls == 1 for path in inputs)

    for mask in ([True, True, False, False], [False, True, True, False]):
        _filter_assembly_paths(paths, np.array(mask, dtype=np.bool_))

    assert all(path.calls == 1 for path in inputs)
