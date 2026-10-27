from pathlib import Path

import numpy as np
import pytest

from seqwin.core import Graph

from _filter_helpers import KMER_DTYPE, _graph


def _build(*args, **kwargs):
    graph = Graph(*args, **kwargs)
    return (
        graph.nodes.kmers, graph.nodes.offsets, graph.nodes.prevalences, graph.nodes.hashes, graph.edges, graph.record_offsets,
        graph.assembly_nodes, graph.node_offsets,
    )


def _small_graph(targets_dir: Path, non_targets_dir: Path) -> Graph:
    return Graph(
        [targets_dir / 'target-1.fasta', non_targets_dir / 'non-target-1.fasta'],
        kmerlen=7,
        windowsize=10,
    )


def test_graph_save_load_round_trip(tmp_path: Path, targets_dir: Path, non_targets_dir: Path) -> None:
    graph = _small_graph(targets_dir, non_targets_dir)
    graph_path = tmp_path / 'graph'
    graph_path.mkdir()
    graph.save(graph_path)

    arrays = _arrays(graph)
    assert {path.name for path in graph_path.iterdir()} == {f'{name}.npy' for name in arrays}
    loaded = _arrays(Graph.load(graph_path))
    for name, array in arrays.items():
        assert loaded[name].dtype == array.dtype
        np.testing.assert_array_equal(loaded[name], array)


def _assert_edges_weight_sorted(edges: np.ndarray) -> None:
    assert np.all(edges['weight'][:-1] >= edges['weight'][1:])
    for weight in np.unique(edges['weight']):
        tied = edges[edges['weight'] == weight]
        endpoints = tied[['first', 'second']].tolist()
        assert endpoints == sorted(endpoints)


def _assert_indexed_edges(nodes: np.ndarray, edges: np.ndarray) -> None:
    assert np.all(edges['first'] < len(nodes))
    assert np.all(edges['second'] < len(nodes))
    assert np.all(edges['first'] <= edges['second'])


def _assert_graph_outputs_equal(expected, actual) -> None:
    for expected_array, actual_array in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(expected_array, actual_array)
    kmers, offsets, prevalences, hashes, edges, *_ = actual
    _assert_edges_weight_sorted(edges)
    _assert_indexed_edges(hashes, edges)
    _assert_node_ranges(kmers, offsets, prevalences, hashes)


def _assert_node_ranges(kmers, offsets, prevalences, hashes) -> None:
    assert kmers.dtype == KMER_DTYPE
    assert offsets.dtype == np.dtype(np.uintp)
    assert prevalences.dtype == np.dtype(np.uint32)
    assert hashes.dtype == np.dtype(np.uint64)
    assert len(offsets) == len(hashes) + 1
    assert len(prevalences) == len(hashes)
    assert offsets[0] == 0
    assert offsets[-1] == len(kmers)
    assert np.all(offsets[:-1] < offsets[1:])
    assert np.all(hashes[:-1] < hashes[1:])
    for start, stop in zip(offsets[:-1], offsets[1:]):
        records = kmers[start:stop]['record_idx']
        assert np.all(records[:-1] <= records[1:])


def test_build_threading_equivalence(targets_dir, non_targets_dir) -> None:
    assembly_paths = [
        targets_dir / 'target-1.fasta',
        targets_dir / 'target-2.fasta',
        non_targets_dir / 'non-target-1.fasta',
        non_targets_dir / 'non-target-2.fasta',
    ]
    single_worker = _build(assembly_paths, kmerlen=7, windowsize=10, n_cpu=1)
    for n_cpu in (2, 99):
        parallel = _build(assembly_paths, kmerlen=7, windowsize=10, n_cpu=n_cpu)
        _assert_graph_outputs_equal(single_worker, parallel)


def test_duplicate_edges_across_workers_are_aggregated(tmp_path: Path) -> None:
    assembly_paths = [tmp_path / f'assembly-{i}.fasta' for i in range(2)]
    for path in assembly_paths:
        path.write_text('>record\nACGTTGCATGTCGCATGATGCATGAGAGCT\n')

    single_worker = Graph(assembly_paths, kmerlen=5, windowsize=6, n_cpu=1)
    multiple_workers = Graph(assembly_paths, kmerlen=5, windowsize=6, n_cpu=2)

    assert len(multiple_workers.edges) > 0
    assert np.all(multiple_workers.edges['weight'] == 2)
    assert np.array_equal(multiple_workers.edges, single_worker.edges)
    _assert_edges_weight_sorted(multiple_workers.edges)
    _assert_indexed_edges(multiple_workers.nodes.hashes, multiple_workers.edges)


def test_multi_worker_record_offsets_and_global_record_indices(tmp_path: Path) -> None:
    def write_fasta(path: Path, n_records: int) -> None:
        seq = 'ACGT' * 20
        path.write_text(''.join(f'>r{i}\n{seq}\n' for i in range(n_records)))

    assembly_paths = []
    for i, n_records in enumerate([2, 1, 3, 1]):
        path = tmp_path / f'a{i}.fasta'
        write_fasta(path, n_records)
        assembly_paths.append(path)

    kmers, _, _, _, _, record_offsets, _, _ = _build(
        assembly_paths,
        kmerlen=7,
        windowsize=10,
        n_cpu=2,
    )

    assert record_offsets.dtype == np.dtype(np.uint32)
    assert np.array_equal(record_offsets, np.array([0, 2, 3, 6, 7], dtype=np.uint32))
    assert np.array_equal(np.unique(kmers['record_idx']), np.arange(7, dtype=np.uint32))


def test_build_empty_record_offsets(tmp_path: Path) -> None:
    empty_assembly = tmp_path / 'empty.fasta'
    empty_assembly.write_text('')

    for assembly_paths, expected_offsets in (
        ([], np.array([0], dtype=np.uint32)),
        ([empty_assembly], np.array([0, 0], dtype=np.uint32)),
        ([empty_assembly] * 3, np.zeros(4, dtype=np.uint32)),
    ):
        for low_memory in (False, True):
            kmers, offsets, prevalences, hashes, _, record_offsets, assembly_nodes, node_offsets = _build(
                assembly_paths,
                kmerlen=7,
                windowsize=10,
                n_cpu=2,
                low_memory=low_memory,
            )
            assert len(kmers) == 0
            _assert_node_ranges(kmers, offsets, prevalences, hashes)
            np.testing.assert_array_equal(offsets, [0])
            assert len(hashes) == 0
            assert len(assembly_nodes) == 0
            assert assembly_nodes.dtype == np.dtype(np.uintp)
            assert np.array_equal(node_offsets, np.zeros(len(assembly_paths) + 1, dtype=np.uintp))
            assert record_offsets.dtype == np.dtype(np.uint32)
            assert np.array_equal(record_offsets, expected_offsets)


def test_low_memory_build_matches_standard(targets_dir, non_targets_dir) -> None:
    assembly_paths = [
        targets_dir / 'target-1.fasta',
        targets_dir / 'target-2.fasta',
        non_targets_dir / 'non-target-1.fasta',
        non_targets_dir / 'non-target-2.fasta',
    ]
    standard = _build(
        assembly_paths, kmerlen=7, windowsize=10, n_cpu=1, low_memory=False,
    )

    for n_cpu in (1, 2, 99):
        low_memory = _build(
            assembly_paths, kmerlen=7, windowsize=10, n_cpu=n_cpu, low_memory=True,
        )
        _assert_graph_outputs_equal(standard, low_memory)


def test_assembly_nodes_match_kmer_memberships(tmp_path: Path) -> None:
    assembly_paths = [tmp_path / f'assembly-{i}.fasta' for i in range(4)]
    assembly_paths[0].write_text('>r0\nACGTACGTACGTACGT\n>r1\nACGTACGTACGTACGT\n')
    assembly_paths[1].write_text('')
    assembly_paths[2].write_text('>short\nAC\n')
    assembly_paths[3].write_text('>r0\nTTTTACGTACGTGGGG\n')

    graph = Graph(assembly_paths, kmerlen=5, windowsize=6, n_cpu=3)
    assert graph.assembly_nodes.dtype == np.dtype(np.uintp)
    assert graph.node_offsets.dtype == np.dtype(np.uintp)
    assert len(graph.node_offsets) == len(assembly_paths) + 1
    assert graph.node_offsets[0] == 0
    assert graph.node_offsets[-1] == len(graph.assembly_nodes)
    assert np.all(graph.node_offsets[:-1] <= graph.node_offsets[1:])
    assert np.all(graph.assembly_nodes < len(graph.nodes.hashes))

    record_assembly = np.empty(int(graph.record_offsets[-1]), dtype=np.uint32)
    for assembly_i in range(len(assembly_paths)):
        record_assembly[
            graph.record_offsets[assembly_i]:graph.record_offsets[assembly_i + 1]
        ] = assembly_i

    expected = [[] for _ in assembly_paths]
    _assert_node_ranges(graph.nodes.kmers, graph.nodes.offsets, graph.nodes.prevalences, graph.nodes.hashes)
    for node_i in range(len(graph.nodes.hashes)):
        start, stop = graph.nodes.offsets[node_i:node_i + 2]
        assemblies = np.unique(record_assembly[
            graph.nodes.kmers['record_idx'][start:stop]
        ])
        assert graph.nodes.prevalences[node_i] == len(assemblies)
        for assembly_i in assemblies:
            expected[int(assembly_i)].append(node_i)

    for assembly_i, expected_nodes in enumerate(expected):
        actual = graph.assembly_nodes[
            graph.node_offsets[assembly_i]:graph.node_offsets[assembly_i + 1]
        ]
        assert np.array_equal(actual, np.asarray(expected_nodes, dtype=np.uintp))
        assert np.all(actual[:-1] < actual[1:])
        assert len(actual) == len(np.unique(actual))


def _arrays(graph):
    return dict(
        kmers=graph.nodes.kmers, offsets=graph.nodes.offsets,
        prevalences=graph.nodes.prevalences, hashes=graph.nodes.hashes,
        edges=graph.edges, record_offsets=graph.record_offsets,
        assembly_nodes=graph.assembly_nodes, node_offsets=graph.node_offsets,
    )


def test_constructor_path_types(targets_dir, non_targets_dir):
    paths = [targets_dir / 'target-1.fasta', non_targets_dir / 'non-target-1.fasta']
    expected = _arrays(Graph(paths, 7, 10))
    for items in (list(map(str, paths)), [str(paths[0]), paths[1]]):
        for name, array in _arrays(Graph(items, 7, 10)).items():
            np.testing.assert_array_equal(array, expected[name])


@pytest.mark.parametrize('mapped', (False, True))
def test_graph_array_ownership_and_readonly(tmp_path, targets_dir, non_targets_dir, mapped, monkeypatch):
    import gc
    import weakref

    graph = _small_graph(targets_dir, non_targets_dir)
    if mapped:
        graph.save(tmp_path)
        original_load = np.load
        mapped_arrays = {}

        def load(path, **kwargs):
            array = original_load(path, **kwargs)
            mapped_arrays[Path(path).stem] = array
            return array

        monkeypatch.setattr(np, 'load', load)
        graph = Graph.load(tmp_path)
    nodes = graph.nodes
    arrays = _arrays(graph)
    expected = {name: array.copy() for name, array in arrays.items()}
    graph_ref = weakref.ref(graph)
    for name, array in arrays.items():
        repeated = _arrays(graph)[name]
        assert np.shares_memory(array, repeated)
        assert not array.flags.writeable
        assert not array.flags.owndata
        if mapped:
            assert array is repeated
            assert array is mapped_arrays[name]
            assert isinstance(array, np.memmap)
            assert array.mode == 'r'
        else:
            assert array.base is graph
        with pytest.raises(ValueError):
            array.setflags(write=True)
        if array.size:
            with pytest.raises(ValueError):
                array[0] = array[0]
    del graph, repeated, array
    gc.collect()
    assert (graph_ref() is None) == mapped
    for name in ('kmers', 'offsets', 'prevalences', 'hashes'):
        np.testing.assert_array_equal(getattr(nodes, name), expected[name])
    del nodes
    gc.collect()
    for name, array in arrays.items():
        np.testing.assert_array_equal(array, expected[name])
    del arrays, array
    gc.collect()
    assert graph_ref() is None  # No ownership cycles.


def test_nodes_alone_retains_graph(targets_dir, non_targets_dir):
    import gc
    graph = _small_graph(targets_dir, non_targets_dir)
    nodes = graph.nodes
    expected = nodes.kmers.copy()
    del graph
    gc.collect()
    np.testing.assert_array_equal(nodes.kmers, expected)


def test_load_old_fixture(expected_graph):
    graph = Graph.load(expected_graph)
    for name, array in _arrays(graph).items():
        expected = np.load(expected_graph / f'{name}.npy')
        np.testing.assert_array_equal(array, expected)


@pytest.mark.parametrize('low_memory', (False, True))
def test_empty_graph_round_trip(tmp_path, low_memory):
    graph = Graph([], 7, 10, low_memory=low_memory)
    graph.save(tmp_path)
    loaded = Graph.load(tmp_path)
    loaded_arrays = _arrays(loaded)
    for name, array in _arrays(graph).items():
        np.testing.assert_array_equal(array, loaded_arrays[name])


@pytest.mark.parametrize(('name', 'array', 'message'), (
    ('prevalences', np.array([2], dtype=np.uint32), 'equal lengths'),
    ('offsets', np.array([0], dtype=np.uintp), 'n_nodes'),
    ('offsets', np.array([1, 2, 4, 8, 9], dtype=np.uintp), 'start at zero'),
    ('offsets', np.array([0, 2, 4, 8, 10], dtype=np.uintp), 'end at len'),
    ('record_offsets', np.array([], dtype=np.uint32), 'must not be empty'),
    ('record_offsets', np.array([1, 1, 2, 3, 4], dtype=np.uint32), 'start at zero'),
    ('node_offsets', np.array([0, 4, 7, 9], dtype=np.uintp), 'equal lengths'),
    ('node_offsets', np.array([0, 4, 7, 8, 10], dtype=np.uintp), 'end at len'),
    ('node_offsets', np.empty(0, dtype=np.uintp), 'must not be empty'),
    ('kmers', np.empty((1, 1), dtype=KMER_DTYPE), 'one-dimensional'),
    ('hashes', np.zeros(4, dtype=np.uint32), 'has dtype'),
    ('hashes', np.arange(4, dtype=np.uint64).astype(np.dtype(np.uint64).newbyteorder('S')), 'has dtype'),
    ('kmers', np.zeros(9, dtype={'names': ['pos', 'record_idx'], 'formats': ['u4', 'u4'],
                                'offsets': [0, 8], 'itemsize': 12}), 'has dtype'),
))
def test_load_rejects_unsafe_array_metadata(tmp_path, name, array, message):
    from _filter_helpers import _inputs

    kmers, offsets, prevalences, edges, record_offsets, _ = _inputs()
    arrays = dict(
        kmers=kmers, offsets=offsets, prevalences=prevalences, edges=edges,
        record_offsets=record_offsets,
        assembly_nodes=np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp),
        node_offsets=np.array([0, 4, 7, 8, 9], dtype=np.uintp),
        hashes=np.arange(4, dtype=np.uint64),
    )
    arrays[name] = array
    with pytest.raises(ValueError, match=message):
        _graph(tmp_path, **arrays)


@pytest.mark.parametrize('layout', ('strided', 'unaligned'))
def test_load_rejects_unsafe_numpy_layout(monkeypatch, expected_graph, layout):
    original_load = np.load
    if layout == 'strided':
        bad = np.zeros(8, dtype=np.uint64)[::2]
        message = 'C-contiguous'
    else:
        bad = np.ndarray((4,), dtype=np.uint64, buffer=bytearray(33), offset=1)
        message = 'aligned'

    def load(path, **kwargs):
        return bad if Path(path).stem == 'hashes' else original_load(path, **kwargs)

    monkeypatch.setattr(np, 'load', load)
    with pytest.raises(ValueError, match=message):
        Graph.load(expected_graph)
