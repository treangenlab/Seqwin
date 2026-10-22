from pathlib import Path

import pytest

from seqwin.config import Config
from seqwin.entry import run


def test_download_only_does_not_execute_full_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    invoked = {'run_called': False}

    def _boom(self):
        invoked['run_called'] = True
        raise AssertionError('Seqwin.run() should not be called in download_only mode')

    monkeypatch.setattr('seqwin.entry.Seqwin.run', _boom)

    config = Config(prefix=tmp_path, title='download-only', download_only=True)
    seq = run(config)

    assert invoked['run_called'] is False
    assert (tmp_path / 'download-only' / 'config.json').exists()
    assert len(seq.assemblies) == 0
    assert seq.signatures is None
    assert seq.metrics is None


def test_output_directory_overwrite_behavior(tmp_path: Path) -> None:
    run_config = dict(
        prefix=tmp_path,
        title='api-lifecycle',
        download_only=True,
    )

    seq = run(Config(**run_config))
    out_dir = tmp_path / 'api-lifecycle'
    assert (out_dir / 'config.json').exists()
    assert seq.state.working_dir == out_dir

    with pytest.raises(FileExistsError):
        run(Config(**run_config))

    rerun = run(Config(**run_config, overwrite=True))
    assert rerun.state.working_dir == out_dir


@pytest.mark.parametrize(('run_blast', 'has_blast'), ((False, True), (True, False), (True, True)))
def test_signature_output_and_evaluation_lifecycle(
    monkeypatch, tmp_path, targets_txt, non_targets_txt, run_blast, has_blast,
):
    import pandas as pd
    from seqwin.core._native import _METADATA_NAMES
    from seqwin.assemblies import Assemblies
    from seqwin.entry import load
    from seqwin.evaluation import SignatureMetrics, _METRIC_NAMES

    monkeypatch.setattr('seqwin.entry.HAS_BLAST', has_blast)
    calls = []

    def database(self, **kwargs):
        calls.append(kwargs)
        return tmp_path / 'all'

    def evaluate(seqs, blastdb, total_tar, total_neg, n_cpu):
        assert blastdb == tmp_path / 'all'
        assert (total_tar, total_neg, n_cpu) == (2, 2, 1)
        return [SignatureMetrics(conservation=.5) for _ in seqs]

    monkeypatch.setattr(Assemblies, 'makeblastdb', database)
    monkeypatch.setattr('seqwin.evaluation.eval_signatures', evaluate)
    seq = run(Config(
        prefix=tmp_path, title='lifecycle', tar_paths=targets_txt,
        neg_paths=non_targets_txt, run_blast=run_blast, kmerlen=17,
        windowsize=10, min_len=17, max_len=200, n_cpu=1, save_pickle=True,
    ))
    output = pd.read_csv(seq.state.working_dir / 'signatures.csv')
    mandatory = _METADATA_NAMES
    assert isinstance(seq.signatures, tuple)
    if run_blast and has_blast:
        assert tuple(output.columns) == (*mandatory, *_METRIC_NAMES)
        assert isinstance(seq.metrics, tuple)
        assert len(seq.metrics) == len(seq.signatures)
        assert calls == [dict(
            prefix=seq.state.working_dir / 'blastdb', neg_only=False,
            overwrite=False, n_cpu=1,
        )]
    else:
        assert tuple(output.columns) == mandatory
        assert seq.metrics is None
        assert not calls
    from seqwin.config import WORKINGDIR
    restored = load(seq.state.working_dir / WORKINGDIR.results)
    assert restored.metrics == seq.metrics
    assert [s.n_tar for s in restored.signatures] == [s.n_tar for s in seq.signatures]
