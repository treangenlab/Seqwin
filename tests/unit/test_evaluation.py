import csv
from pathlib import Path

import numpy as np
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


def test_hits_keep_optimal_alignment_and_input_order(monkeypatch):
    # Query 0 has no hits. Query 1 has repeats and tied non-target bitscores.
    tar = next(k for k, v in BLASTCONFIG.str2bool.items() if v)
    neg = next(k for k, v in BLASTCONFIG.str2bool.items() if not v)
    hits = np.array([
        (2, BLASTCONFIG.header_sep.join(['0', tar, 'other']), 15, 3, 1, 0),
        (1, BLASTCONFIG.header_sep.join(['0', tar, 'record']), 10, 2, 1, 0),
        (1, BLASTCONFIG.header_sep.join(['2', neg, 'first']), 20, 3, 1, 1),
        (1, BLASTCONFIG.header_sep.join(['0', tar, 'record']), 20, 4, 0, 0),
        (1, BLASTCONFIG.header_sep.join(['2', neg, 'tied']), 20, 1, 3, 0),
        (1, BLASTCONFIG.header_sep.join(['1', tar, 'record']), 10, 2, 2, 0),
        (1, BLASTCONFIG.header_sep.join(['3', neg, 'record']), 10, 2, 1, 0),
    ], dtype=[
        ('qseqid', np.int64), ('sseqid', object), ('bitscore', np.float64),
        ('nident', np.int64), ('mismatch', np.int64), ('gaps', np.int64),
    ])
    monkeypatch.setattr(evaluation, 'blast', lambda *args, **kwargs: hits.copy())
    metrics = evaluation.eval_signatures(['AAAA', 'CCCC', 'GGGG'], Path(BLASTCONFIG.title_all), 3, 3, 1)
    assert metrics[0].blast.size == 0
    assert all(getattr(metrics[0], name) == 0.0 for name in evaluation._METRIC_NAMES)
    assert metrics[1].conservation == .5
    assert metrics[1].f_tar_hits == pytest.approx(2 / 3)
    assert metrics[1].avg_repeats_tar == 1.5
    assert metrics[1].avg_pident_tar == .625
    assert metrics[1].divergence == .25
    assert metrics[1].f_neg_hits == pytest.approx(2 / 3)
    assert metrics[1].avg_repeats_neg == 1.5
    assert metrics[1].avg_pident_neg == .5
    assert metrics[1].blast['bitscore'].tolist() == [20, 10, 20, 10]
    assert metrics[1].blast['record_id'].tolist() == ['record', 'record', 'first', 'record']
    assert metrics[1].blast['n_hits'].tolist() == [2, 1, 2, 1]
    assert metrics[1].blast['avg_nident'].tolist() == [3, 2, 2, 2]
    assert metrics[2].conservation == .25
    assert metrics[2].blast['record_id'].tolist() == ['other']


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
