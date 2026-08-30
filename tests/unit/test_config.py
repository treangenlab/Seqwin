from pathlib import Path

import pytest
from pydantic import ValidationError

from seqwin.config import Config


@pytest.mark.parametrize('mode', ('paths', 'dirs'))
def test_config_resolves_input_paths(
    request: pytest.FixtureRequest, tmp_path: Path, mode: str,
) -> None:
    fixture_suffix = 'txt' if mode == 'paths' else 'dir'
    field_suffix = 'paths' if mode == 'paths' else 'dir'
    tar = request.getfixturevalue(f'targets_{fixture_suffix}')
    neg = request.getfixturevalue(f'non_targets_{fixture_suffix}')
    config = Config(**{f'tar_{field_suffix}': tar, f'neg_{field_suffix}': neg, 'prefix': tmp_path})

    assert getattr(config, f'tar_{field_suffix}') == tar.resolve(strict=True)
    assert getattr(config, f'neg_{field_suffix}') == neg.resolve(strict=True)
    assert config.prefix == tmp_path.resolve(strict=True)


def test_download_only_allows_no_inputs(tmp_path: Path) -> None:
    config = Config(prefix=tmp_path, download_only=True)
    assert config.download_only is True


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
