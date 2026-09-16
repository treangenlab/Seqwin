from pathlib import Path

import numpy as np
import pytest

import seqwin.markers as markers
from seqwin.assemblies import Assemblies
from seqwin.graph import EDGE_DTYPE, KMER_DTYPE, NODE_DTYPE
from seqwin.graph import _extract_native
from seqwin.kmers import FilteredGraph
from seqwin.config import CONSEC_KMER_MUL


KMERLEN = 5
WINDOWSIZE = 10


def _write_assemblies(
    tmp_path: Path, records_by_assembly: list[list[str]], is_targets: list[bool]
) -> Assemblies:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for assembly_idx, records in enumerate(records_by_assembly):
        path = tmp_path / f'assembly-{assembly_idx}.fasta'
        path.write_text(''.join(
            f'>record-{record_idx}\n{sequence}\n'
            for record_idx, sequence in enumerate(records)
        ))
        paths.append(path)
    return Assemblies(paths, is_targets)


def _synthetic_graph(
    records_by_assembly: list[list[str]],
    subgraphs: list[list[tuple[int, int, int, int]]],
) -> FilteredGraph:
    """Build a graph from subgraphs of (hash, assembly, record, position)."""
    record_offsets = np.concatenate((
        np.array([0], dtype=np.uint32),
        np.cumsum([len(records) for records in records_by_assembly], dtype=np.uint32),
    ))
    occurrences_by_hash = {}
    hashes_by_subgraph = []
    for occurrences in subgraphs:
        subgraph_hashes = []
        for kmer_hash, assembly_idx, record_idx, position in occurrences:
            if kmer_hash not in occurrences_by_hash:
                occurrences_by_hash[kmer_hash] = []
            if kmer_hash not in subgraph_hashes:
                subgraph_hashes.append(kmer_hash)
            occurrences_by_hash[kmer_hash].append((
                position, int(record_offsets[assembly_idx]) + record_idx,
            ))
        hashes_by_subgraph.append(subgraph_hashes)

    kmers = []
    nodes = []
    node_idx = {}
    for kmer_hash, occurrences in occurrences_by_hash.items():
        start = len(kmers)
        kmers.extend(occurrences)
        node_idx[kmer_hash] = len(nodes)
        nodes.append((kmer_hash, start, len(kmers), 0, 0, 0.0))

    return FilteredGraph(
        kmers=np.array(kmers, dtype=KMER_DTYPE),
        nodes=np.array(nodes, dtype=NODE_DTYPE),
        edges=np.array([], dtype=EDGE_DTYPE),
        record_offsets=record_offsets,
        record_ids=np.array([
            f'record-{record_idx}'
            for records in records_by_assembly
            for record_idx in range(len(records))
        ]),
        subgraphs=[
            [node_idx[kmer_hash] for kmer_hash in hashes]
            for hashes in hashes_by_subgraph
        ],
    )


def _extract(
    tmp_path: Path,
    records_by_assembly: list[list[str]],
    is_targets: list[bool],
    subgraphs: list[list[tuple[int, int, int, int]]],
    *,
    n_cpu: int = 1,
    min_len: int = 0,
):
    assemblies = _write_assemblies(tmp_path, records_by_assembly, is_targets)
    graph = _synthetic_graph(records_by_assembly, subgraphs)
    return markers._get_cks(
        graph, sum(is_targets), KMERLEN, WINDOWSIZE, min_len, assemblies, n_cpu,
    )


def _ck_fields(ck):
    return {
        'assembly_idx': int(ck.rep['assembly_idx']),
        'record_idx': int(ck.rep['record_idx']),
        'start': int(ck.rep['start']),
        'stop': int(ck.rep['stop']),
        'n_kmers': int(ck.rep['n_kmers']),
        'kmers': tuple(ck.rep['kmers']),
        'n_repeats': int(ck.rep['n_repeats']),
        'seq': ck.rep['seq'],
        'len': int(ck.len),
        'n_rep': int(ck.n_rep),
        'rep_ratio': ck.rep_ratio,
        'warnings': ck.warnings,
        'is_bad': ck.is_bad,
    }


def _native_fields(signature):
    loc = signature.location
    return {
        'assembly_idx': loc.assembly_idx,
        'record_idx': loc.record_idx,
        'start': loc.start,
        'stop': loc.stop,
        'n_kmers': loc.n_kmers,
        'n_repeats': loc.n_repeats,
        'seq': signature.sequence,
        'len': signature.length,
        'n_rep': signature.n_rep,
        'rep_ratio': signature.rep_ratio,
    }


def _extract_native_fixture(
    graph, assemblies, *, min_len=0, n_cpu=1,
    total_tar=None, consec_kmer_mul=CONSEC_KMER_MUL,
):
    if total_tar is None:
        total_tar = int(assemblies.is_targets.sum())
    return _extract_native(
        graph.kmers, graph.nodes, graph.subgraphs, graph.record_offsets,
        assemblies.is_targets, [str(path) for path in assemblies.paths],
        KMERLEN, WINDOWSIZE, min_len, total_tar, consec_kmer_mul, n_cpu,
    )


def _python_parity_fields(ck):
    fields = _ck_fields(ck)
    fields.pop('kmers')
    fields.pop('warnings')
    fields.pop('is_bad')
    return fields


def test_multiple_fasta_records_do_not_form_one_run(tmp_path: Path) -> None:
    records = [['A' * 100, 'CCGGTTAAACCCGGTTAAAA']]
    subgraphs = [[
        (11, 0, 0, 90),
        (12, 0, 1, 2),
        (13, 0, 1, 7),
    ]]

    cks, sequences = _extract(tmp_path, records, [True], subgraphs)

    assert sequences == ['GGTTAAACCC']
    assert _ck_fields(cks[0]) == {
        'assembly_idx': 0, 'record_idx': 1, 'start': 2, 'stop': 12,
        'n_kmers': 2, 'kmers': (12, 13), 'n_repeats': 2,
        'seq': 'GGTTAAACCC', 'len': 10, 'n_rep': 1, 'rep_ratio': 1.0,
        'warnings': set(), 'is_bad': False,
    }


def test_largest_of_separated_repeated_runs_is_selected(tmp_path: Path) -> None:
    sequence = 'ACGT' * 30
    subgraphs = [[
        (1, 0, 0, 2), (2, 0, 0, 6), (3, 0, 0, 10),
        (1, 0, 0, 50), (2, 0, 0, 54),
    ]]

    cks, sequences = _extract(tmp_path, [[sequence]], [True], subgraphs)

    assert sequences == [sequence[2:15]]
    assert _ck_fields(cks[0]) == {
        'assembly_idx': 0, 'record_idx': 0, 'start': 2, 'stop': 15,
        'n_kmers': 3, 'kmers': (1, 2, 3), 'n_repeats': 2,
        'seq': sequence[2:15], 'len': 13, 'n_rep': 1, 'rep_ratio': 1.0,
        'warnings': set(), 'is_bad': False,
    }


def test_equal_size_run_tie_selects_first_sorted_run(tmp_path: Path) -> None:
    records = [['A' * 30, 'CGTACGTACGTACGTACGTA']]
    subgraphs = [[
        (1, 0, 0, 20), (2, 0, 0, 24),
        (1, 0, 1, 3), (2, 0, 1, 7),
    ]]

    cks, sequences = _extract(tmp_path, records, [True], subgraphs)

    assert sequences == ['A' * 9]
    assert (cks[0].rep['record_idx'], cks[0].rep['start']) == (0, 20)
    assert (cks[0].rep['stop'], cks[0].rep['n_kmers'], cks[0].rep['n_repeats']) == (29, 2, 2)


def test_forward_reverse_orders_share_class_and_canonical_orientation_wins_tie(tmp_path: Path) -> None:
    records = [['AACCGGTTAACCGGTT'], ['TTTTCCCCAAAAGGGG']]
    subgraphs = [[
        (3, 0, 0, 1), (2, 0, 0, 5), (1, 0, 0, 9),
        (1, 1, 0, 2), (2, 1, 0, 6), (3, 1, 0, 10),
    ]]

    cks, sequences = _extract(tmp_path, records, [True, True], subgraphs)

    assert sequences == ['TTCCCCAAAAGGG']
    assert tuple(cks[0].rep['kmers']) == (1, 2, 3)
    assert (cks[0].rep['assembly_idx'], cks[0].rep['start'], cks[0].rep['stop']) == (1, 2, 15)
    assert (cks[0].n_rep, cks[0].rep_ratio) == (2, 1.0)


def test_equal_canonical_scores_select_first_encountered_order(tmp_path: Path) -> None:
    records = [
        ['AAAACCCCGGGGTTTTAAAA'],
        ['TTTTGGGGCCCCAAAA'],
        ['ACGTACGTACGTACGT'],
    ]
    subgraphs = [[
        (1, 0, 0, 1), (2, 0, 0, 5), (3, 0, 0, 9), (4, 0, 0, 13),
        (5, 1, 0, 2), (6, 1, 0, 6),
        (5, 2, 0, 3), (6, 2, 0, 7),
    ]]

    cks, sequences = _extract(tmp_path, records, [True, True, True], subgraphs)

    # Scores tie at 4: four k-mers in one target versus two k-mers in two targets.
    assert sequences == ['AAACCCCGGGGTTTTAA']
    assert tuple(cks[0].rep['kmers']) == (1, 2, 3, 4)
    assert (cks[0].rep['assembly_idx'], cks[0].rep['record_idx'], cks[0].rep['start']) == (0, 0, 1)
    assert (cks[0].n_rep, cks[0].rep_ratio) == (1, 1 / 3)


def test_representative_location_uses_first_matching_assembly(tmp_path: Path) -> None:
    records = [
        ['AAAACCCCGGGGTTTTAAAA'],
        ['TTTTGGGGCCCCAAAATTTT'],
    ]
    subgraphs = [[
        (10, 0, 0, 4), (11, 0, 0, 8), (12, 0, 0, 12),
        (10, 1, 0, 2), (11, 1, 0, 6), (12, 1, 0, 10),
    ]]

    cks, sequences = _extract(tmp_path, records, [True, True], subgraphs)

    assert tuple(cks[0].rep['kmers']) == (10, 11, 12)
    assert (cks[0].rep['assembly_idx'], cks[0].rep['start'], cks[0].rep['stop']) == (0, 4, 17)
    assert sequences == ['CCCCGGGGTTTTA']
    assert (cks[0].n_rep, cks[0].rep_ratio) == (2, 1.0)


def test_min_len_includes_equal_length_and_excludes_shorter(tmp_path: Path) -> None:
    sequence = 'AAAACCCCGGGGTTTT'
    subgraphs = [
        [(20, 0, 0, 1), (21, 0, 0, 5)],  # length 9
        [(30, 0, 0, 3), (31, 0, 0, 6)],  # length 8
    ]

    cks, sequences = _extract(
        tmp_path, [[sequence]], [True], subgraphs, min_len=9,
    )

    assert len(cks) == 1
    assert tuple(cks[0].rep['kmers']) == (20, 21)
    assert (cks[0].rep['start'], cks[0].rep['stop'], cks[0].len) == (1, 10, 9)
    assert sequences == ['AAACCCCGG']


def test_bad_duplicate_and_single_signatures_are_excluded(tmp_path: Path) -> None:
    records = [['ACGT' * 30]]
    duplicate = [(7, 0, 0, 1), (8, 0, 0, 5), (7, 0, 0, 9)]
    single = [(9, 0, 0, 20)]
    valid = [(10, 0, 0, 30), (11, 0, 0, 34)]
    graph = _synthetic_graph(records, [duplicate, single, valid])
    assemblies = _write_assemblies(tmp_path, records, [True])

    raw = [markers._create_ck(*args) for args in markers._get_create_ck_args(
        graph, assemblies, KMERLEN, WINDOWSIZE,
    )]
    assert [(ck.warnings, ck.is_bad) for ck in raw] == [
        ({'dup'}, True), ({'single'}, True), (set(), False),
    ]

    cks, sequences = markers._get_cks(
        graph, 1, KMERLEN, WINDOWSIZE, 0, assemblies, 1,
    )
    assert len(cks) == 1
    assert tuple(cks[0].rep['kmers']) == (10, 11)
    assert sequences == [(records[0][0])[30:39]]


def test_candidate_order_follows_subgraph_order(tmp_path: Path) -> None:
    sequence = 'AAAACCCCGGGGTTTT' * 5
    subgraphs = [
        [(30, 0, 0, 20), (31, 0, 0, 24)],
        [(10, 0, 0, 2), (11, 0, 0, 6), (12, 0, 0, 10)],
    ]

    cks, sequences = _extract(tmp_path, [[sequence]], [True], subgraphs)

    assert [tuple(ck.rep['kmers']) for ck in cks] == [(30, 31), (10, 11, 12)]
    assert [int(ck.rep['start']) for ck in cks] == [20, 2]
    assert sequences == [sequence[20:29], sequence[2:15]]


def test_thread_count_preserves_candidate_order_and_fields(tmp_path: Path) -> None:
    records = [
        ['AAAACCCCGGGGTTTT' * 5],
        ['TTTTGGGGCCCCAAAA' * 5],
        ['ACGT' * 20],
    ]
    subgraphs = [
        [(20, 0, 0, 3), (21, 0, 0, 7), (20, 1, 0, 4), (21, 1, 0, 8)],
        [(30, 1, 0, 30), (31, 1, 0, 34), (30, 2, 0, 5), (31, 2, 0, 9)],
    ]

    serial, serial_sequences = _extract(
        tmp_path / 'serial', records, [True, False, True], subgraphs, n_cpu=1,
    )
    parallel, parallel_sequences = _extract(
        tmp_path / 'parallel', records, [True, False, True], subgraphs, n_cpu=2,
    )

    assert serial_sequences == parallel_sequences
    assert [_ck_fields(ck) for ck in serial] == [_ck_fields(ck) for ck in parallel]
    assert [tuple(ck.rep['kmers']) for ck in serial] == [(20, 21), (30, 31)]


def test_create_ck_uses_interleaved_target_mask(monkeypatch) -> None:
    captured = dict()

    def capture_connected_kmers(kmers, kmerlen, windowsize):
        captured['kmers'] = kmers
        return object()

    monkeypatch.setattr(markers, 'ConnectedKmers', capture_connected_kmers)
    kmers = np.array(
        [(10, 0), (20, 1), (30, 2), (40, 3)],
        dtype=KMER_DTYPE
    )
    is_targets = np.array([False, True, False, True], dtype=np.bool_)

    markers._create_ck(
        (np.uint64(1),),
        (kmers,),
        np.array([0, 1, 2, 3, 4], dtype=np.uint32),
        is_targets,
        7,
        10
    )

    result = captured['kmers'].sort_values('assembly_idx')
    assert np.array_equal(result['assembly_idx'], np.arange(4))
    assert np.array_equal(result['record_idx'], np.zeros(4))
    assert np.array_equal(result['is_target'], is_targets)


def test_create_ck_args_slices_non_contiguous_raw_kmer_ranges() -> None:
    kmers = np.array(
        [(10, 0), (11, 1), (99, 0), (20, 0), (21, 1)],
        dtype=KMER_DTYPE,
    )
    graph = FilteredGraph(
        kmers=kmers,
        nodes=np.array(
            [(10, 0, 2, 0, 0, 0.0), (20, 3, 5, 0, 0, 0.0)],
            dtype=NODE_DTYPE,
        ),
        edges=np.array([(0, 1, 1)], dtype=EDGE_DTYPE),
        record_offsets=np.array([0, 1, 2], dtype=np.uint32),
        record_ids=np.array(['a', 'b']),
        subgraphs=[[0, 1],],
    )
    assemblies = type('AssembliesStub', (), {
        'is_targets': np.array([True, False], dtype=np.bool_),
    })()

    args = next(markers._get_create_ck_args(graph, assemblies, 7, 10))
    groups_by_hash = dict(zip(args[0], args[1]))

    np.testing.assert_array_equal(groups_by_hash[np.uint64(10)], kmers[0:2])
    np.testing.assert_array_equal(groups_by_hash[np.uint64(20)], kmers[3:5])


def test_native_extraction_matches_characterized_python_behavior(tmp_path: Path) -> None:
    records = [
        ['AAAACCCCGGGGTTTTAAAA' * 3],
        ['TTTTGGGGCCCCAAAATTTT' * 3],
        ['ACGT' * 30],
    ]
    subgraphs = [
        # Equal weighted canonical scores select the first encountered order.
        [
            (1, 0, 0, 1), (2, 0, 0, 5), (3, 0, 0, 9), (4, 0, 0, 13),
            (5, 1, 0, 2), (6, 1, 0, 6),
            (5, 2, 0, 3), (6, 2, 0, 7),
        ],
        # Matching locations select the first target assembly.
        [
            (10, 0, 0, 24), (11, 0, 0, 28), (12, 0, 0, 32),
            (10, 1, 0, 22), (11, 1, 0, 26), (12, 1, 0, 30),
        ],
        [(20, 0, 0, 40), (21, 0, 0, 44), (20, 0, 0, 48)],  # duplicate
        [(30, 0, 0, 50)],  # single
        [(40, 0, 0, 55), (41, 0, 0, 56)],  # shorter than min_len
    ]
    assemblies = _write_assemblies(tmp_path, records, [True, True, True])
    graph = _synthetic_graph(records, subgraphs)
    python, _ = markers._get_cks(
        graph, 3, KMERLEN, WINDOWSIZE, 9, assemblies, 1,
    )

    serial = _extract_native_fixture(graph, assemblies, min_len=9, n_cpu=1)
    parallel = _extract_native_fixture(graph, assemblies, min_len=9, n_cpu=3)

    assert [signature.subgraph_idx for signature in serial] == [0, 1]
    assert len(serial) == len(python)
    assert [_native_fields(signature) for signature in serial] == [
        _python_parity_fields(ck) for ck in python
    ]
    assert [_native_fields(signature) for signature in parallel] == [
        _native_fields(signature) for signature in serial
    ]
    assert [signature.subgraph_idx for signature in parallel] == [0, 1]


def test_native_run_scanning_boundaries_and_ties(tmp_path: Path) -> None:
    records = [['A' * 100, 'CGTACGTACGTACGTACGTA']]
    subgraphs = [
        # The final k-mer ends sg_kmers and the later separated repeat is shorter.
        [
            (1, 0, 0, 2), (2, 0, 0, 6), (3, 0, 0, 10),
            (1, 0, 0, 50), (2, 0, 0, 54),
        ],
        # Equal runs do not cross records and the first sorted run wins.
        [
            (10, 0, 0, 80), (11, 0, 0, 84),
            (10, 0, 1, 3), (11, 0, 1, 7),
        ],
    ]
    assemblies = _write_assemblies(tmp_path, records, [True])
    graph = _synthetic_graph(records, subgraphs)

    signatures = _extract_native_fixture(graph, assemblies)

    assert [signature.subgraph_idx for signature in signatures] == [0, 1]
    assert _native_fields(signatures[0]) == {
        'assembly_idx': 0, 'record_idx': 0, 'start': 2, 'stop': 15,
        'n_kmers': 3, 'n_repeats': 2, 'seq': 'A' * 13,
        'len': 13, 'n_rep': 1, 'rep_ratio': 1.0,
    }
    assert (signatures[1].location.record_idx, signatures[1].location.start) == (0, 80)
    assert signatures[1].location.n_repeats == 2


def test_native_extract_config_controls_target_ratio_and_run_gap(tmp_path: Path) -> None:
    records = [['ACGT' * 30]]
    subgraphs = [[(1, 0, 0, 2), (2, 0, 0, 14)]]
    assemblies = _write_assemblies(tmp_path, records, [True])
    graph = _synthetic_graph(records, subgraphs)

    # A smaller multiplier splits the occurrences into single-k-mer runs, which
    # makes the candidate invalid; the supplied larger multiplier groups them.
    assert _extract_native_fixture(
        graph, assemblies, consec_kmer_mul=1.0,
    ) == []
    grouped = _extract_native_fixture(
        graph, assemblies, consec_kmer_mul=1.2,
    )
    assert len(grouped) == 1
    assert (grouped[0].location.n_kmers, grouped[0].location.n_repeats) == (2, 1)

    with pytest.raises(ValueError, match='at least one target'):
        _extract_native_fixture(graph, assemblies, total_tar=0)
