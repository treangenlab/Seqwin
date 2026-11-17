import csv
from pathlib import Path

import numpy as np
import pytest

import seqwin.evaluation as evaluation
from seqwin.config import BLASTCONFIG
from seqwin.core._native import _METADATA_NAMES
from seqwin.evaluation import SignatureMetrics, eval_signatures_seqwin


COLUMNS = _METADATA_NAMES


def _files(tmp_path):
    fasta = tmp_path / 'signatures.fasta'
    csv_path = tmp_path / 'signatures.csv'
    fasta.write_text('>second\nAAAA\n>first\nCCCC\n')
    with csv_path.open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(COLUMNS)
        writer.writerows([
            ('second', 4, .95, 3, 1), ('first', 4, .85, 2, 2),
        ])
    return fasta, csv_path


def test_file_evaluation_preserves_order_and_appends_metrics(monkeypatch, tmp_path):
    fasta, csv_path = _files(tmp_path)
    metrics = [SignatureMetrics(conservation=.1), SignatureMetrics(conservation=.8)]

    def evaluate(seqs, db, tar, neg, cpu):
        assert (seqs, db, tar, neg, cpu) == (['AAAA', 'CCCC'], Path('db'), 2, 3, 1)
        return metrics

    monkeypatch.setattr(evaluation, 'eval_signatures', evaluate)
    assert eval_signatures_seqwin(fasta, csv_path, Path('db'), 2, 3) == metrics
    with csv_path.open(newline='') as source:
        reader = csv.DictReader(source)
        output = list(reader)
    assert tuple(reader.fieldnames) == (*COLUMNS, *evaluation._METRIC_NAMES)
    assert [row['fasta_header'] for row in output] == ['second', 'first']
    assert [float(row['conservation']) for row in output] == [.1, .8]


@pytest.mark.parametrize(('change', 'message'), [
    ('count', 'counts'), ('header', 'header mismatch at index 0'),
    ('length', 'length mismatch at index 0'),
])
def test_file_mismatches_fail_before_evaluation(monkeypatch, tmp_path, change, message):
    fasta, csv_path = _files(tmp_path)
    with csv_path.open(newline='') as source:
        rows = list(csv.DictReader(source))
    if change == 'count':
        rows = rows[:1]
    else:
        rows[0]['fasta_header' if change == 'header' else 'length'] = 'wrong' if change == 'header' else 5
    with csv_path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    before = csv_path.read_bytes()
    monkeypatch.setattr(evaluation, 'eval_signatures', lambda *args: pytest.fail('unexpected evaluation'))
    with pytest.raises(ValueError, match=message):
        eval_signatures_seqwin(fasta, csv_path, Path('db'), 1, 1)
    assert csv_path.read_bytes() == before


def test_failed_evaluation_preserves_csv(monkeypatch, tmp_path):
    fasta, csv_path = _files(tmp_path)
    before = csv_path.read_bytes()

    def fail(*args):
        raise RuntimeError('BLAST failed')

    monkeypatch.setattr(evaluation, 'eval_signatures', fail)
    with pytest.raises(RuntimeError, match='BLAST failed'):
        eval_signatures_seqwin(fasta, csv_path, Path('db'), 1, 1)
    assert csv_path.read_bytes() == before


def test_hits_keep_optimal_alignment_and_input_order(monkeypatch):
    # Query 0 has no hits. Query 1 has repeats and tied non-target bitscores.
    # Query 2 has only target hits. Query 3 has only non-target hits.
    tar = next(k for k, v in BLASTCONFIG.str2bool.items() if v)
    neg = next(k for k, v in BLASTCONFIG.str2bool.items() if not v)
    hits = np.zeros(8, dtype=evaluation._BLAST_DTYPE)
    hits[['qseqid', 'sseqid', 'bitscore', 'nident', 'mismatch', 'gaps']] = [
        (2, BLASTCONFIG.header_sep.join(['0', tar, 'other']), 15, 3, 1, 0),
        (1, BLASTCONFIG.header_sep.join(['0', tar, 'record']), 10, 2, 1, 0),
        (1, BLASTCONFIG.header_sep.join(['2', neg, 'first']), 20, 3, 1, 1),
        (1, BLASTCONFIG.header_sep.join(['0', tar, 'record']), 20, 4, 0, 0),
        (1, BLASTCONFIG.header_sep.join(['2', neg, 'tied']), 20, 1, 3, 0),
        (1, BLASTCONFIG.header_sep.join(['1', tar, 'record']), 10, 2, 2, 0),
        (1, BLASTCONFIG.header_sep.join(['3', neg, 'record']), 10, 2, 1, 0),
        (3, BLASTCONFIG.header_sep.join(['2', neg, 'negative']), 10, 2, 1, 1),
    ]
    monkeypatch.setattr(evaluation, 'blast', lambda *args, **kwargs: hits.copy())
    metrics = evaluation.eval_signatures(('AAAA', 'CCCC', 'GGGG', 'TTTT'), Path(BLASTCONFIG.title_all), 3, 3, 1)
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
    assert metrics[3].conservation == 0.0
    assert metrics[3].f_tar_hits == 0.0
    assert metrics[3].avg_repeats_tar == 0.0
    assert metrics[3].avg_pident_tar == 0.0
    assert metrics[3].divergence == pytest.approx(1 / 6)
    assert metrics[3].f_neg_hits == pytest.approx(1 / 3)
    assert metrics[3].avg_repeats_neg == 1
    assert metrics[3].avg_pident_neg == .5
    assert metrics[3].blast['record_id'].tolist() == ['negative']


def test_empty_blast_output(monkeypatch):
    monkeypatch.setattr(evaluation, 'blast', lambda *args, **kwargs: np.empty(0, dtype=evaluation._BLAST_DTYPE))
    seq_list = ('AAAA', 'CCCC')
    metrics = evaluation.eval_signatures(seq_list, Path(BLASTCONFIG.title_all), 1, 1)
    assert len(metrics) == len(seq_list)
    default = SignatureMetrics()
    for metric in [*metrics, default]:
        assert all(getattr(metric, name) == 0.0 for name in evaluation._METRIC_NAMES)
        assert metric.blast.size == 0
        assert metric.blast.dtype == evaluation._BEST_DTYPE
    assert default.blast is not SignatureMetrics().blast


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
