from pathlib import Path

import pytest

from seqwin.utils import read_text


@pytest.fixture(scope='session')
def smoke_root() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope='session')
def fixtures_dir(smoke_root: Path) -> Path:
    return smoke_root / 'fixtures'


@pytest.fixture(scope='session')
def targets_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / 'targets'


@pytest.fixture(scope='session')
def non_targets_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / 'non-targets'


@pytest.fixture(scope='session')
def targets_txt(fixtures_dir: Path) -> Path:
    return fixtures_dir / 'targets.txt'


@pytest.fixture(scope='session')
def non_targets_txt(fixtures_dir: Path) -> Path:
    return fixtures_dir / 'non-targets.txt'


@pytest.fixture(scope='session')
def expected_fasta(fixtures_dir: Path) -> str:
    return read_text(fixtures_dir / 'expected' / 'signatures.fasta')
