from pathlib import Path

import pytest
from pydantic import ValidationError

from seqwin.config import Config


def test_config_from_txt_paths(tmp_path: Path, targets_txt: Path, non_targets_txt: Path) -> None:
    config = Config(
        tar_paths=targets_txt,
        neg_paths=non_targets_txt,
        prefix=tmp_path,
        run_mash=False,
    )

    assert config.tar_paths == targets_txt.resolve(strict=True)
    assert config.neg_paths == non_targets_txt.resolve(strict=True)
    assert config.prefix == tmp_path.resolve(strict=True)


def test_config_from_dirs(tmp_path: Path, targets_dir: Path, non_targets_dir: Path) -> None:
    config = Config(
        tar_dir=targets_dir,
        neg_dir=non_targets_dir,
        prefix=tmp_path,
        run_mash=False,
    )

    assert config.tar_dir == targets_dir.resolve(strict=True)
    assert config.neg_dir == non_targets_dir.resolve(strict=True)


def test_download_only_allows_no_inputs(tmp_path: Path) -> None:
    config = Config(prefix=tmp_path, download_only=True)
    assert config.download_only is True


def test_invalid_values_raise_validation_error(tmp_path: Path, targets_txt: Path, non_targets_txt: Path) -> None:
    common = dict(tar_paths=targets_txt, neg_paths=non_targets_txt, prefix=tmp_path)

    with pytest.raises(ValidationError, match='penalty_th must be between'):
        Config(**common, penalty_th=1.5)

    with pytest.raises(ValidationError, match='stringency must be between'):
        Config(**common, stringency=11)

    with pytest.raises(ValidationError, match='max_len must be greater than min_len'):
        Config(**common, min_len=50, max_len=50)


def test_missing_inputs_raise_when_not_download_only(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match='You must provide at least one target input'):
        Config(prefix=tmp_path)


def test_config_is_frozen(tmp_path: Path, targets_txt: Path, non_targets_txt: Path) -> None:
    config = Config(tar_paths=targets_txt, neg_paths=non_targets_txt, prefix=tmp_path)

    with pytest.raises(ValidationError, match='Instance is frozen'):
        config.n_cpu = 1  # type: ignore[misc]


def test_json_serialization_contains_important_fields(
    tmp_path: Path,
    targets_txt: Path,
    non_targets_txt: Path,
) -> None:
    config = Config(
        tar_paths=targets_txt,
        neg_paths=non_targets_txt,
        prefix=tmp_path,
        run_mash=False,
        run_blast=False,
        n_cpu=2,
    )

    json_text = config.model_dump_json()

    assert '"version"' in json_text
    assert '"prefix"' in json_text
    assert '"tar_paths"' in json_text
    assert '"neg_paths"' in json_text
    assert '"run_mash":false' in json_text
    assert '"run_blast":false' in json_text
    assert '"n_cpu":2' in json_text
