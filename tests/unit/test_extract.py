import csv

import numpy as np
import pytest

from seqwin.core import EDGE_DTYPE, KMER_DTYPE
from seqwin.core._native import _METADATA_NAMES, _filter_native

from _filter_helpers import _filter, _inputs, _paths


def _extract_signature(tmp_path, *, target_records, edge_data):
    """One record per target assembly, followed by a non-target with no nodes."""
    records = [*target_records, []]
    hashes = sorted({node_hash for record in records for node_hash, _ in record})
    kmers, offsets, prevalence, assembly_nodes = [], [], [], []
    node_offsets = [0]
    for node_hash in hashes:
        occurrences = [
            (pos, record_idx)
            for record_idx, record in enumerate(records)
            for value, pos in record if value == node_hash
        ]
        prevalence.append(len({record_idx for _, record_idx in occurrences}))
        offsets.append(len(kmers))
        kmers.extend(occurrences)
    for record in records:
        present = {node_hash for node_hash, _ in record}
        assembly_nodes.extend(i for i, node_hash in enumerate(hashes) if node_hash in present)
        node_offsets.append(len(assembly_nodes))
    _, signatures = _filter_native(
        kmers=np.array(kmers, dtype=KMER_DTYPE),
        offsets=np.array([*offsets, len(kmers)], dtype=np.uintp),
        prevalence=np.array(prevalence, dtype=np.uint32),
        edges=np.array(edge_data, dtype=EDGE_DTYPE),
        record_offsets=np.arange(len(records) + 1, dtype=np.uint32),
        assembly_nodes=np.array(assembly_nodes, dtype=np.uintp),
        node_offsets=np.array(node_offsets, dtype=np.uintp),
        assembly_paths=_paths(tmp_path, len(records), sequence_length=64),
        is_targets=np.array([True] * len(target_records) + [False], dtype=np.bool_),
        jaccard=None, kmerlen=5, windowsize=5, penalty_th=.9, stringency=5,
        min_len=0, max_len=None, penalty_th_cap=.2, edge_w_th_mul=0,
        min_nodes_floor=1, max_nodes_cap=None, consec_kmer_mul=2, n_cpu=1,
    )
    signature, = signatures
    return signature


def test_signature_selects_longest_consecutive_run(tmp_path):
    # A,B then A,B,C: the 25-base gap exceeds windowsize * consec_kmer_mul (10).
    record = [(10, 0), (20, 5), (10, 30), (20, 35), (30, 40)]
    signature = _extract_signature(
        tmp_path, target_records=[record, record],
        edge_data=[(0, 1, 4), (1, 2, 2)],
    )
    assert signature.location.start == 30
    assert signature.location.stop == 40 + 5
    assert signature.location.n_kmers == 3
    assert signature.location.n_repeats == 2


def test_signature_reverse_order_canonicalization(tmp_path):
    # A,B,C and C,B,A share support; the latter is the majority orientation.
    signature = _extract_signature(
        tmp_path,
        target_records=[
            [(10, 0), (20, 5), (30, 10)],
            [(30, 10), (20, 15), (10, 20)],
            [(30, 0), (20, 5), (10, 10)],
        ],
        edge_data=[(0, 1, 3), (1, 2, 3)],
    )
    assert signature.n_tar == 3
    assert signature.location.assembly_idx == 1
    assert signature.location.record_idx == 0
    assert signature.location.start == 10
    assert signature.location.stop == 20 + 5


def test_signature_representative_score_tie(tmp_path):
    # B,A,C and A,B,C each score 3 minimizers * 2 targets.
    # Assembly 0 encounters B,A,C first, although its canonical order sorts later.
    first = [(20, 10), (10, 15), (30, 20)]
    second = [(10, 0), (20, 5), (30, 10)]
    signature = _extract_signature(
        tmp_path, target_records=[first, second, first, second],
        edge_data=[(0, 1, 4), (0, 2, 2), (1, 2, 2)],
    )
    assert signature.location.assembly_idx == 0
    assert signature.location.record_idx == 0
    assert signature.location.start == 10
    assert signature.location.stop == 20 + 5
    assert signature.n_tar == 2


def test_signature_extraction_uses_final_original_node_range(tmp_path):
    (filtered, signatures), _ = _filter(
        tmp_path,
        penalty_th=2.0, edge_w_th_mul=0, min_nodes_floor=1,
    )

    assert filtered.nodes['idx'][-1] == 3
    assert any(3 in subgraph for subgraph in filtered.subgraphs)
    assert signatures


@pytest.mark.parametrize('final_start_offset', (0, 1))
def test_signature_extraction_rejects_invalid_final_node_range(tmp_path, final_start_offset):
    kmers, offsets, prevalence, edges, record_offsets, targets = _inputs()
    offsets[-2] = len(kmers) + final_start_offset
    assembly_nodes = np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp)
    node_offsets = np.array([0, 4, 7, 8, 9], dtype=np.uintp)

    with pytest.raises(ValueError, match='Node k-mer range is invalid'):
        _filter_native(
            kmers, offsets, prevalence, edges, record_offsets, assembly_nodes, node_offsets,
            _paths(tmp_path), targets, None, 5, 10, 2.0, 5, 0, None, .2, 0, 1,
            None, 1.5, 1,
        )


@pytest.mark.parametrize('record_id', ('ordinary', 'record,with"quotes'))
def test_native_csv_escaping_and_fasta_wrapping(tmp_path, record_id):
    kmers, offsets, prevalence, edges, record_offsets, targets = _inputs()
    paths = []
    for i in range(4):
        path = tmp_path / f'{i}.fasta'
        path.write_text(f'>{record_id}\n' + 'acgt' * 50 + '\n')
        paths.append(str(path))
    fasta_path = tmp_path / 'signatures.fasta'
    csv_path = tmp_path / 'signatures.csv'
    _filter_native(
        kmers, offsets, prevalence, edges, record_offsets,
        np.array([0, 1, 2, 3, 0, 1, 2, 2, 2], dtype=np.uintp),
        np.array([0, 4, 7, 8, 9], dtype=np.uintp),
        paths, targets, None, 100, 10, .3, 5, 0, None, .2, .3, 1, None, 1.5, 1,
        fasta_path=str(fasta_path), csv_path=str(csv_path),
    )
    header = f'0-{record_id}-0:100'
    assert fasta_path.read_text().splitlines() == ['>' + header, 'ACGT' * 20, 'ACGT' * 5]
    with csv_path.open(newline='') as source:
        rows = list(csv.reader(source))
    assert rows == [list(_METADATA_NAMES), [header, '100', '1', '2', '1']]
    assert csv_path.read_text().splitlines()[0] == ','.join(_METADATA_NAMES)
    expected = header if record_id == 'ordinary' else '"' + header.replace('"', '""') + '"'
    assert csv_path.read_text().splitlines()[1] == expected + ',100,1,2,1'
