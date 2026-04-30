from pathlib import Path

from typer.testing import CliRunner

from seqwin import cli
from seqwin.config import WORKINGDIR
from seqwin.utils import read_text

runner = CliRunner()

_shared_config = (
    '--kmerlen', '17',
    '--windowsize', '10',
    '--min-len', '17',
    '--max-len', '200',
    '--no-mash',
)


def _run_cli(*args: str) -> Path:
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.stdout

    prefix = Path(args[args.index('--prefix') + 1])
    title = args[args.index('--title') + 1] if '--title' in args else 'seqwin-out'
    out_dir = prefix / title
    assert out_dir.exists()
    return out_dir


def test_txt_mode_blast_on_matches_expected(tmp_path: Path, targets_txt: Path, non_targets_txt: Path, expected_fasta_blast: str) -> None:
    out_dir = _run_cli(
        '--tar-paths', str(targets_txt),
        '--neg-paths', str(non_targets_txt),
        '--prefix', str(tmp_path),
        '--threads', '1',
        '--title', 'txt-blast-on',
        *_shared_config,
    )

    assert read_text(out_dir / WORKINGDIR.markers_fasta) == expected_fasta_blast


def test_txt_mode_blast_off_matches_expected(tmp_path: Path, targets_txt: Path, non_targets_txt: Path, expected_fasta: str) -> None:
    out_dir = _run_cli(
        '--tar-paths', str(targets_txt),
        '--neg-paths', str(non_targets_txt),
        '--prefix', str(tmp_path),
        '--threads', '1',
        '--title', 'txt-blast-off',
        '--no-blast',
        *_shared_config,
    )

    assert read_text(out_dir / WORKINGDIR.markers_fasta) == expected_fasta


def test_txt_vs_dir_mode_match(
    tmp_path: Path,
    targets_txt: Path, non_targets_txt: Path,
    targets_dir: Path, non_targets_dir: Path,
    expected_fasta: str,
) -> None:
    test_config = (
        '--prefix', str(tmp_path),
        '--no-blast',
        '--threads', '1',
    )
    txt_out = _run_cli(
        '--tar-paths', str(targets_txt),
        '--neg-paths', str(non_targets_txt),
        '--title', 'txt-mode',
        *test_config,
        *_shared_config,
    )
    dir_out = _run_cli(
        '--tar-dir', str(targets_dir),
        '--neg-dir', str(non_targets_dir),
        '--title', 'dir-mode',
        *test_config,
        *_shared_config,
    )

    assert read_text(txt_out / WORKINGDIR.markers_fasta) == read_text(dir_out / WORKINGDIR.markers_fasta) == expected_fasta


def test_threads_one_vs_two_match(tmp_path: Path, targets_txt: Path, non_targets_txt: Path, expected_fasta: str) -> None:
    test_config = (
        '--tar-paths', str(targets_txt),
        '--neg-paths', str(non_targets_txt),
        '--prefix', str(tmp_path),
        '--no-blast',
    )
    one_out = _run_cli(
        *test_config,
        '--threads', '1',
        '--title', 'threads-1',
        *_shared_config,
    )
    two_out = _run_cli(
        *test_config,
        '--threads', '2',
        '--title', 'threads-2',
        *_shared_config,
    )

    assert read_text(one_out / WORKINGDIR.markers_fasta) == read_text(two_out / WORKINGDIR.markers_fasta) == expected_fasta
