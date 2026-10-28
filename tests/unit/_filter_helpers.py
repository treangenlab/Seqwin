import numpy as np

from seqwin.core import Graph
from seqwin.core._native import _filter_native


KMER_DTYPE = np.dtype([('pos', np.uint32), ('record_idx', np.uint32)])
EDGE_DTYPE = np.dtype([('first', np.uintp), ('second', np.uintp), ('weight', np.uintp)])


def _graph(kmers, offsets, prevalences, edges, record_offsets,
           assembly_nodes, node_offsets, *, hashes=None):
    """Construct a synthetic graph from read-only NumPy arrays."""
    if hashes is None:
        hashes = np.arange(len(prevalences), dtype=np.uint64)
    arrays = dict(
        kmers=kmers, offsets=offsets, prevalences=prevalences, hashes=hashes,
        edges=edges, record_offsets=record_offsets, assembly_nodes=assembly_nodes,
        node_offsets=node_offsets,
    )
    for array in arrays.values():
        array.setflags(write=False)
    return Graph._from_numpy(**arrays)


def _paths(tmp_path, n_assemblies=4, *, sequence_length=12):
    paths = []
    for i in range(n_assemblies):
        path = tmp_path / f'assembly-{i}.fasta'
        sequence = ('ACGT' * sequence_length)[:sequence_length]
        path.write_text(f'>record-{i}\n{sequence}\n')
        paths.append(str(path))
    return paths


def _inputs():
    occurrences = {10: (0, 1), 20: (0, 1), 30: (0, 1, 2, 3), 40: (0,)}
    kmers = []
    offsets = []
    prevalences = []
    for node_hash, records in occurrences.items():
        start = len(kmers)
        kmers.extend((i, record) for i, record in enumerate(records))
        offsets.append(start)
        prevalences.append(len(set(records)))
    return (
        np.array(kmers, dtype=KMER_DTYPE),
        np.array([*offsets, len(kmers)], dtype=np.uintp),
        np.array(prevalences, dtype=np.uint32),
        np.array([(0, 1, 1), (1, 2, 1), (2, 3, 1)], dtype=EDGE_DTYPE),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
        np.array([True, True, False, False], dtype=np.bool_),
    )


def _filter(tmp_path, *, penalty_th=0.3, jaccard=None, n_cpu=1, targets=None,
            penalty_th_cap=0.2, edge_w_th_mul=0.3, min_nodes_floor=1,
            max_nodes_cap=None, **output_paths):
    kmers, offsets, prevalences, edges, record_offsets, default_targets = _inputs()
    if targets is None:
        targets = default_targets
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)
    graph = _graph(
        kmers, offsets, prevalences, edges, record_offsets, assembly_nodes, node_offsets,
    )
    result = _filter_native(
        graph,
        _paths(tmp_path), targets, jaccard, 5, 10,
        penalty_th, 5, 0, None, penalty_th_cap, edge_w_th_mul, min_nodes_floor,
        max_nodes_cap, 1.5, n_cpu, **output_paths,
    )
    return result, prevalences
