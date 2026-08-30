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
    assert len(seq.assemblies) == 0


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
