from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import seqwin.evaluation as evaluation
from seqwin.evaluation import SignatureMetrics, process_signatures


class Location:
    assembly_idx = 0
    record_idx = 1
    start = 2
    stop = 8
    n_kmers = 3
    n_repeats = 1


def _signature(index=0, sequence='ACGTAC'):
    return SimpleNamespace(
        subgraph_idx=index, location=Location(), sequence=sequence,
        length=len(sequence), n_rep=1, rep_ratio=1.0
    )


def test_signature_metrics_stores_blast_separately():
    blast = pd.DataFrame({'nident': [4]})
    metric = SignatureMetrics(conservation=1.0, blast=blast)
    assert metric.blast is blast
    assert 'blast' not in evaluation._METRIC_NAMES


def test_private_process_signatures_writes_record_id_and_outputs(tmp_path: Path):
    signature = _signature()
    filtered = SimpleNamespace(total_tar=1, total_neg=1)
    config = SimpleNamespace(
        overwrite=False, run_blast=False, blast_neg_only=False, n_cpu=1
    )
    state = SimpleNamespace(working_dir=tmp_path, blastdb=None)
    assemblies = SimpleNamespace()

    graph = SimpleNamespace(
        record_offsets=np.array([0, 2], dtype=np.uint32),
        record_ids=np.array(['first', 'second'])
    )
    processed = process_signatures(
        [signature], filtered, assemblies, graph, config, state
    )

    signatures, metrics = processed
    assert signatures == (signature,)
    assert metrics == (evaluation.SignatureMetrics(),)
    assert (tmp_path / 'signatures.fasta').read_text() == '>0-second-2:8\nACGTAC\n'
    output = pd.read_csv(tmp_path / 'signatures.csv')
    assert tuple(output.columns) == (
        'fasta_header', 'length', *evaluation._METRIC_NAMES, 'rep_ratio', 'n_nodes'
    )
    assert output.loc[0, 'fasta_header'] == '0-second-2:8'
    assert output.loc[0, 'length'] == 6
    assert output.loc[0, 'rep_ratio'] == 1.0
    assert output.loc[0, 'n_nodes'] == 3


def test_evaluation_attaches_results_and_ranks(monkeypatch):
    first = _signature(0, 'AAAA')
    second = _signature(1, 'CCCC')
    low = evaluation.SignatureMetrics(conservation=.1, divergence=.2)
    high = evaluation.SignatureMetrics(conservation=.8, divergence=.1)
    monkeypatch.setattr(evaluation, 'eval_signatures', lambda *args: [low, high])

    signatures = [first, second]
    ranked_signatures, ranked_metrics = evaluation._eval_signatures(
        signatures, Path('all'), 1, 1, 1
    )

    assert ranked_signatures == [second, first]
    assert ranked_metrics == [high, low]
    assert signatures == [first, second]
