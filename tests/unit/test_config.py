from pathlib import Path

import pytest
from pydantic import ValidationError

from seqwin.config import Config


@pytest.mark.parametrize(
    'invalid',
    (
        pytest.param({'penalty_th': 1.5}, id='penalty-threshold'),
        pytest.param({'stringency': 11}, id='stringency'),
        pytest.param({'min_len': 51, 'max_len': 50}, id='length-order'),
    ),
)
def test_invalid_values_raise_validation_error(
    tmp_path: Path, targets_txt: Path, non_targets_txt: Path, invalid: dict,
) -> None:
    common = dict(tar_paths=targets_txt, neg_paths=non_targets_txt, prefix=tmp_path)

    with pytest.raises(ValidationError):
        Config(**common, **invalid)


def test_missing_inputs_raise_when_not_download_only(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Config(prefix=tmp_path)


def test_config_is_frozen(tmp_path: Path, targets_txt: Path, non_targets_txt: Path) -> None:
    config = Config(tar_paths=targets_txt, neg_paths=non_targets_txt, prefix=tmp_path)

    with pytest.raises(ValidationError):
        config.n_cpu = 1  # type: ignore[misc]


def test_json_serialization_contains_important_fields_and_redacts_api_key(
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
        low_memory=True,
        api_key='test-key',
    )

    json_text = config.model_dump_json()

    assert '"version"' in json_text
    assert '"prefix"' in json_text
    assert '"tar_paths"' in json_text
    assert '"neg_paths"' in json_text
    assert '"run_mash":false' in json_text
    assert '"run_blast":false' in json_text
    assert '"n_cpu":2' in json_text
    assert '"low_memory":true' in json_text
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == 'test-key'
    assert '"api_key":"**********"' in json_text
    assert 'test-key' not in json_text
