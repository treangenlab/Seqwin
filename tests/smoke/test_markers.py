import networkx as nx
import numpy as np

import seqwin.markers as markers
from seqwin.graph import KMER_DTYPE


def test_create_ck_uses_interleaved_target_mask(monkeypatch) -> None:
    captured = dict()

    def capture_connected_kmers(graph, kmers, kmerlen, windowsize):
        captured['kmers'] = kmers
        return object()

    monkeypatch.setattr(markers, 'ConnectedKmers', capture_connected_kmers)
    kmers = np.array(
        [(10, 0), (20, 1), (30, 2), (40, 3)],
        dtype=KMER_DTYPE
    )
    is_targets = np.array([False, True, False, True], dtype=np.bool_)

    markers._create_ck(
        nx.Graph(),
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
