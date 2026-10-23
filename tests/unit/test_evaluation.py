import csv
from pathlib import Path

import pandas as pd
import pytest

import seqwin.evaluation as evaluation
from seqwin.config import BLASTCONFIG
from seqwin.core._native import _METADATA_NAMES
from seqwin.evaluation import SignatureMetrics, eval_signatures_seqwin


COLUMNS = _METADATA_NAMES


def _files(tmp_path):
    fasta = tmp_path / 'signatures.fasta'
    csv = tmp_path / 'signatures.csv'
    fasta.write_text('>second\nAAAA\n>first\nCCCC\n')
    pd.DataFrame([
        ('second', 4, .95, 3, 1), ('first', 4, .85, 2, 2),
    ], columns=COLUMNS).to_csv(csv, index=False)
    return fasta, csv


def test_file_evaluation_preserves_order_and_appends_metrics(monkeypatch, tmp_path):
    fasta, csv = _files(tmp_path)
    metrics = [SignatureMetrics(conservation=.1), SignatureMetrics(conservation=.8)]

    def evaluate(seqs, db, tar, neg, cpu):
        assert (seqs, db, tar, neg, cpu) == (['AAAA', 'CCCC'], Path('db'), 2, 3, 1)
        return metrics

    monkeypatch.setattr(evaluation, 'eval_signatures', evaluate)
    assert eval_signatures_seqwin(fasta, csv, Path('db'), 2, 3) == metrics
    output = pd.read_csv(csv)
    assert tuple(output.columns) == (*COLUMNS, *evaluation._METRIC_NAMES)
    assert output.fasta_header.tolist() == ['second', 'first']
    assert output.conservation.tolist() == [.1, .8]


@pytest.mark.parametrize(('change', 'message'), [
    ('count', 'counts'), ('header', 'header mismatch at index 0'),
    ('length', 'length mismatch at index 0'),
])
def test_file_mismatches_fail_before_evaluation(monkeypatch, tmp_path, change, message):
    fasta, csv = _files(tmp_path)
    df = pd.read_csv(csv)
    if change == 'count':
        df = df.iloc[:1]
    else:
        df.loc[0, 'fasta_header' if change == 'header' else 'length'] = 'wrong' if change == 'header' else 5
    df.to_csv(csv, index=False)
    before = csv.read_bytes()
    monkeypatch.setattr(evaluation, 'eval_signatures', lambda *args: pytest.fail('unexpected evaluation'))
    with pytest.raises(ValueError, match=message):
        eval_signatures_seqwin(fasta, csv, Path('db'), 1, 1)
    assert csv.read_bytes() == before


def test_failed_evaluation_preserves_csv(monkeypatch, tmp_path):
    fasta, csv = _files(tmp_path)
    before = csv.read_bytes()

    def fail(*args):
        raise RuntimeError('BLAST failed')

    monkeypatch.setattr(evaluation, 'eval_signatures', fail)
    with pytest.raises(RuntimeError, match='BLAST failed'):
        eval_signatures_seqwin(fasta, csv, Path('db'), 1, 1)
    assert csv.read_bytes() == before


@pytest.mark.parametrize('n_cpu', (1, 2))
def test_hits_keep_optimal_alignment_and_input_order(monkeypatch, n_cpu):
    # Query 0 has no hits. Query 1 has two alignments in the same assembly.
    hits = pd.DataFrame([
        (1, BLASTCONFIG.header_sep.join(['0', next(k for k, v in BLASTCONFIG.str2bool.items() if v), 'record']), 10, 2, 1, 0),
        (1, BLASTCONFIG.header_sep.join(['0', next(k for k, v in BLASTCONFIG.str2bool.items() if v), 'record']), 20, 4, 0, 0),
    ], columns=['qseqid', 'sseqid', 'bitscore', 'nident', 'mismatch', 'gaps'])
    monkeypatch.setattr(evaluation, 'blast', lambda *args, **kwargs: hits.copy())
    metrics = evaluation.eval_signatures(['AAAA', 'CCCC'], Path(BLASTCONFIG.title_all), 1, 1, n_cpu)
    assert metrics[0].blast is None
    assert all(getattr(metrics[0], name) == 0.0 for name in evaluation._METRIC_NAMES)
    assert metrics[1].conservation == 1.0
    assert metrics[1].avg_repeats_tar == 2
    assert metrics[1].avg_pident_tar == .75
    assert metrics[1].blast.bitscore.tolist() == [20]


def test_csv_rewrite_preserves_metadata_strings_and_escaping(monkeypatch, tmp_path):
    fasta, path = _files(tmp_path)
    header = 'record,with"quotes'
    fasta.write_text(f'>{header}\nAAAA\n')
    row = [header, '04', '0.33333333333333331', '03', '01']
    with path.open('w', newline='') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(COLUMNS)
        writer.writerow(row)
    monkeypatch.setattr(evaluation, 'eval_signatures', lambda *args: [SignatureMetrics()])
    eval_signatures_seqwin(fasta, path, Path('db'), 1, 1)
    with path.open(newline='') as source:
        rows = list(csv.reader(source))
    assert rows[1][:5] == row
    assert path.read_text().startswith(','.join(COLUMNS) + ',')
    assert '"record,with""quotes",04,0.33333333333333331,03,01,' in path.read_text()
