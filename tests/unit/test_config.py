from pathlib import Path

import pytest
from pydantic import ValidationError

from seqwin.config import Config


@pytest.mark.parametrize(
    'invalid',
    (
        pytest.param({'tar_weight': 11}, id='tar_weight'),
        pytest.param({'min_len': 51, 'max_len': 50}, id='length-order'),
    ),
)
def test_invalid_values_raise_validation_error(
    tmp_path: Path, targets_txt: Path, non_targets_txt: Path, invalid: dict,
) -> None:
    common = dict(tar_paths=targets_txt, neg_paths=non_targets_txt, prefix=tmp_path)

    with pytest.raises(ValidationError):
        Config(**common, **invalid)


@pytest.mark.parametrize('name', ('tar_weight', 'neg_rank'))
@pytest.mark.parametrize('value', (-.01, 1.01, float('nan'), float('inf'), float('-inf')))
def test_invalid_prevalence_parameters(targets_txt, non_targets_txt, name, value):
    with pytest.raises(ValidationError, match=name):
        Config(tar_paths=targets_txt, neg_paths=non_targets_txt, **{name: value})


@pytest.mark.parametrize('value', (0.0, 0.37, 1.0))
def test_valid_prevalence_parameters(targets_txt, non_targets_txt, value):
    config = Config(tar_paths=targets_txt, neg_paths=non_targets_txt, tar_weight=value, neg_rank=value)
    assert config.tar_weight == config.neg_rank == value


@pytest.mark.parametrize('name', ('tar_weigth', 'stringency', 'neg_quantile'))
def test_unknown_parameters_are_forbidden(targets_txt, non_targets_txt, name):
    with pytest.raises(ValidationError) as error:
        Config(tar_paths=targets_txt, neg_paths=non_targets_txt, **{name: .5})

    assert error.value.errors()[0]['type'] == 'extra_forbidden'
    assert error.value.errors()[0]['loc'] == (name,)


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
    assert '"n_tar_th":null' in json_text
    assert '"n_neg_th":null' in json_text
    assert '"neg_rank":0.2' in json_text
    assert '"tar_weight":0.5' in json_text
    assert '"run_blast":false' in json_text
    assert '"n_cpu":2' in json_text
    assert '"low_memory":true' in json_text
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == 'test-key'
    assert '"api_key":"**********"' in json_text
    assert 'test-key' not in json_text
