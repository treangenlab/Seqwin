from pathlib import Path

import pytest

from seqwin.config import Config
from seqwin.core import run


def test_download_only_does_not_execute_full_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    invoked = {'run_called': False}

    def _boom(self):
        invoked['run_called'] = True
        raise AssertionError('Seqwin.run() should not be called in download_only mode')

    monkeypatch.setattr('seqwin.core.Seqwin.run', _boom)

    config = Config(prefix=tmp_path, title='download-only', download_only=True)
    seq = run(config)

    assert invoked['run_called'] is False
    assert (tmp_path / 'download-only' / 'config.json').exists()
    assert seq.assemblies.empty


def test_api_run_and_overwrite_behavior(
    tmp_path: Path,
    targets_txt: Path,
    non_targets_txt: Path,
) -> None:
    run_config = dict(
        tar_paths=targets_txt,
        neg_paths=non_targets_txt,
        prefix=tmp_path,
        title='api-smoke',
        run_mash=False,
        run_blast=False,
        kmerlen=7,
        windowsize=10,
        min_len=8,
        max_len=120,
        n_cpu=1,
    )

    seq = run(Config(**run_config))
    out_dir = tmp_path / 'api-smoke'
    assert (out_dir / 'config.json').exists()
    assert (out_dir / 'assemblies.csv').exists()
    assert (out_dir / 'signatures.fasta').exists()
    assert seq.config.run_mash is False

    with pytest.raises(FileExistsError):
        run(Config(**run_config))

    rerun = run(Config(**run_config, overwrite=True))
    assert (out_dir / 'results.seqwin').exists()
    assert rerun.state.working_dir == out_dir
