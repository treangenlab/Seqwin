from pathlib import Path

import numpy as np
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
    '--no-blast',
)


def _assert_graph_matches_expected(actual_path: Path, expected_path: Path) -> None:
    actual = np.load(actual_path, allow_pickle=False)
    expected = np.load(expected_path, allow_pickle=False)

    actual_keys = set(actual.files)
    expected_keys = set(expected.files)
    required_keys = {'kmers', 'nodes', 'edges', 'record_offsets'}

    assert actual_keys == expected_keys, f'graph arrays mismatch: actual={sorted(actual_keys)}, expected={sorted(expected_keys)}'
    assert actual_keys == required_keys, f'graph arrays must be exactly {sorted(required_keys)}, got {sorted(actual_keys)}'

    for name in sorted(required_keys):
        actual_array = actual[name]
        expected_array = expected[name]
        assert actual_array.dtype == expected_array.dtype, f'{name} dtype mismatch: actual={actual_array.dtype}, expected={expected_array.dtype}'
        assert actual_array.shape == expected_array.shape, f'{name} shape mismatch: actual={actual_array.shape}, expected={expected_array.shape}'

    np.testing.assert_array_equal(actual['kmers'], expected['kmers'], err_msg='kmers array values mismatch')
    np.testing.assert_array_equal(actual['edges'], expected['edges'], err_msg='edges array values mismatch')

    nodes_dtype = actual['nodes'].dtype
    for field_name in nodes_dtype.names or ():
        if field_name == 'penalty':
            np.testing.assert_allclose(
                actual['nodes'][field_name],
                expected['nodes'][field_name],
                rtol=0,
                atol=1e-12,
                err_msg='nodes.penalty values mismatch',
            )
        else:
            np.testing.assert_array_equal(
                actual['nodes'][field_name],
                expected['nodes'][field_name],
                err_msg=f'nodes.{field_name} values mismatch',
            )


def _run_cli(*args: str) -> Path:
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.stdout

    prefix = Path(args[args.index('--prefix') + 1])
    title = args[args.index('--title') + 1] if '--title' in args else 'seqwin-out'
    out_dir = prefix / title
    assert out_dir.exists()
    return out_dir


def test_txt_mode_matches_expected(tmp_path: Path, targets_txt: Path, non_targets_txt: Path, expected_fasta: str) -> None:
    out_dir = _run_cli(
        '--tar-paths', str(targets_txt),
        '--neg-paths', str(non_targets_txt),
        '--prefix', str(tmp_path),
        '--threads', '1',
        '--title', 'txt-mode',
        *_shared_config,
    )

    assert read_text(out_dir / WORKINGDIR.markers_fasta) == expected_fasta


def test_dir_mode_matches_expected(tmp_path: Path, targets_dir: Path, non_targets_dir: Path, expected_fasta: str) -> None:
    out_dir = _run_cli(
        '--tar-dir', str(targets_dir),
        '--neg-dir', str(non_targets_dir),
        '--prefix', str(tmp_path),
        '--threads', '1',
        '--title', 'dir-mode',
        *_shared_config,
    )

    assert read_text(out_dir / WORKINGDIR.markers_fasta) == expected_fasta


def test_multithreading_matches_expected(tmp_path: Path, targets_txt: Path, non_targets_txt: Path, expected_fasta: str) -> None:
    out_dir = _run_cli(
        '--tar-paths', str(targets_txt),
        '--neg-paths', str(non_targets_txt),
        '--prefix', str(tmp_path),
        '--threads', '2',
        '--title', 'threads-2',
        *_shared_config,
    )

    assert read_text(out_dir / WORKINGDIR.markers_fasta) == expected_fasta


def test_graph_matches_expected(tmp_path: Path, targets_txt: Path, non_targets_txt: Path, expected_graph: Path) -> None:
    out_dir = _run_cli(
        '--tar-paths', str(targets_txt),
        '--neg-paths', str(non_targets_txt),
        '--prefix', str(tmp_path),
        '--threads', '1',
        '--title', 'no-filter',
        '--no-filter',
        *_shared_config,
    )

    _assert_graph_matches_expected(out_dir / WORKINGDIR.graph, expected_graph)


def test_low_memory_graph_matches_expected(tmp_path: Path, targets_txt: Path, non_targets_txt: Path, expected_graph: Path) -> None:
    out_dir = _run_cli(
        '--tar-paths', str(targets_txt),
        '--neg-paths', str(non_targets_txt),
        '--prefix', str(tmp_path),
        '--threads', '1',
        '--title', 'no-filter-low-memory',
        '--no-filter',
        '--low-memory',
        *_shared_config,
    )

    _assert_graph_matches_expected(out_dir / WORKINGDIR.graph, expected_graph)
