import numpy as np

from seqwin.core import EDGE_DTYPE, KMER_DTYPE, NODE_DTYPE
from seqwin.core._native import _filter_native


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
    nodes = []
    for node_hash, records in occurrences.items():
        start = len(kmers)
        kmers.extend((i, record) for i, record in enumerate(records))
        nodes.append((node_hash, start, len(set(records))))
    return (
        np.array(kmers, dtype=KMER_DTYPE),
        np.array(nodes, dtype=NODE_DTYPE),
        np.array([(0, 1, 1), (1, 2, 1), (2, 3, 1)], dtype=EDGE_DTYPE),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
        np.array([True, True, False, False], dtype=np.bool_),
    )


def _filter(tmp_path, *, penalty_th=0.3, jaccard=None, n_cpu=1, targets=None,
            penalty_th_cap=0.2, edge_w_th_mul=0.3, min_nodes_floor=1,
            max_nodes_cap=None, **output_paths):
    kmers, nodes, edges, offsets, default_targets = _inputs()
    if targets is None:
        targets = default_targets
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)
    result = _filter_native(
        kmers, nodes, edges, offsets, assembly_nodes, node_offsets,
        _paths(tmp_path), targets, jaccard, 5, 10,
        penalty_th, 5, 0, None, penalty_th_cap, edge_w_th_mul, min_nodes_floor,
        max_nodes_cap, 1.5, n_cpu, **output_paths,
    )
    return result, nodes
