from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import seqwin.markers as markers
from seqwin.kmers import FilterResult, Signature


class Location:
    assembly_idx = 0
    record_idx = 1
    start = 2
    stop = 8
    n_kmers = 3
    n_repeats = 1


def test_filter_result_and_signature_data_model():
    signature = Signature(2, Location(), 'ACGTAC', 6, 3, .75)
    result = FilterResult(np.array([]), np.array([]), [[0]], [signature], None)
    assert FilterResult.__slots__ == ('nodes', 'edges', 'subgraphs', 'signatures', 'jaccard')
    assert Signature.__slots__ == (
        'subgraph_idx', 'location', 'sequence', 'length', 'n_rep', 'rep_ratio',
        'blast', 'metrics'
    )
    assert result.signatures == [signature]
    assert result.jaccard is None
    assert signature.blast is None and signature.metrics is None


def test_process_signatures_writes_record_id_and_outputs(tmp_path: Path):
    signature = Signature(0, Location(), 'ACGTAC', 6, 1, 1.0)
    result = FilterResult(np.array([]), np.array([]), [[0]], [signature], None)
    config = SimpleNamespace(
        overwrite=False, run_blast=False, blast_neg_only=False, n_cpu=1
    )
    state = SimpleNamespace(working_dir=tmp_path, total_tar=1, total_neg=1)
    assemblies = SimpleNamespace()

    processed = markers.process_signatures(
        result, np.array([0, 2], dtype=np.uint32),
        np.array(['first', 'second']), assemblies, config, state
    )

    assert processed == [signature]
    assert signature.metrics == markers.SignatureMetrics()
    assert (tmp_path / 'signatures.fasta').read_text() == '>0-second-2:8\nACGTAC\n'
    output = pd.read_csv(tmp_path / 'signatures.csv')
    assert output.loc[0, 'fasta_header'] == '0-second-2:8'
    assert output.loc[0, 'length'] == 6
    assert output.loc[0, 'rep_ratio'] == 1.0
    assert output.loc[0, 'n_nodes'] == 3


def test_evaluation_attaches_results_and_ranks(monkeypatch):
    first = Signature(0, Location(), 'AAAA', 4, 1, 1.0)
    second = Signature(1, Location(), 'CCCC', 4, 1, 1.0)
    low = markers.SignatureMetrics(conservation=.1, divergence=.2)
    high = markers.SignatureMetrics(conservation=.8, divergence=.1)
    blasts = [object(), object()]
    monkeypatch.setattr(markers, 'eval_signatures', lambda *args: (blasts, [low, high]))

    signatures = [first, second]
    markers._eval_signatures(signatures, Path('all'), 1, 1, 1)

    assert signatures == [second, first]
    assert first.blast is blasts[0] and first.metrics is low
    assert second.blast is blasts[1] and second.metrics is high
