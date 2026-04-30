from pathlib import Path

from typer.testing import CliRunner

from seqwin import __version__
from seqwin import cli

runner = CliRunner()


def test_help_shows_key_options() -> None:
    result = runner.invoke(cli.app, ['--help'])

    assert result.exit_code == 0
    for opt in (
        '--tar-paths',
        '--neg-paths',
        '--tar-dir',
        '--neg-dir',
        '--no-mash',
        '--no-blast',
        '--threads',
        '--prefix',
    ):
        assert opt in result.output


def test_version_prints_package_version() -> None:
    result = runner.invoke(cli.app, ['--version'])

    assert result.exit_code == 0
    assert f'Seqwin v{__version__}' in result.output


def test_missing_required_inputs_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ['--prefix', str(tmp_path), '--no-mash'])

    assert result.exit_code != 0
    assert 'You must provide at least one target input' in result.output


def test_cli_to_config_mapping_txt(monkeypatch, tmp_path: Path, targets_txt: Path, non_targets_txt: Path) -> None:
    captured = {}

    def _fake_run(config):
        captured['config'] = config
        return object()

    monkeypatch.setattr('seqwin.cli.run', _fake_run)

    result = runner.invoke(
        cli.app,
        [
            '--tar-paths', str(targets_txt),
            '--neg-paths', str(non_targets_txt),
            '--no-mash',
            '--no-blast',
            '--threads', '2',
            '--prefix', str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    cfg = captured['config']
    assert cfg.run_mash is False
    assert cfg.run_blast is False
    assert cfg.n_cpu == 2
    assert cfg.prefix == tmp_path.resolve(strict=True)
    assert cfg.tar_paths == targets_txt.resolve(strict=True)
    assert cfg.neg_paths == non_targets_txt.resolve(strict=True)


def test_cli_to_config_mapping_dir(monkeypatch, tmp_path: Path, targets_dir: Path, non_targets_dir: Path) -> None:
    captured = {}

    def _fake_run(config):
        captured['config'] = config
        return object()

    monkeypatch.setattr('seqwin.cli.run', _fake_run)

    result = runner.invoke(
        cli.app,
        [
            '--tar-dir', str(targets_dir),
            '--neg-dir', str(non_targets_dir),
            '--no-mash',
            '--threads', '1',
            '--prefix', str(tmp_path),
            '--title', 'cli-dir-mode',
        ],
    )

    assert result.exit_code == 0
    cfg = captured['config']
    assert cfg.tar_dir == targets_dir.resolve(strict=True)
    assert cfg.neg_dir == non_targets_dir.resolve(strict=True)
    assert cfg.prefix == tmp_path.resolve(strict=True)
    assert cfg.title == 'cli-dir-mode'
