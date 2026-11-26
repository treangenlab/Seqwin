import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from seqwin.config import Config


def test_invalid_prevalence_parameters(targets_txt, non_targets_txt):
    with pytest.raises(ValidationError) as error:
        Config(tar_paths=targets_txt, neg_paths=non_targets_txt, tar_weight=-.01, neg_rank=1.01)

    assert {e['loc'] for e in error.value.errors()} == {('tar_weight',), ('neg_rank',)}


def test_valid_prevalence_parameters(targets_txt, non_targets_txt):
    config = Config(tar_paths=targets_txt, neg_paths=non_targets_txt, tar_weight=0, neg_rank=1)
    assert (config.tar_weight, config.neg_rank) == (0, 1)


def test_unknown_parameters_are_forbidden(targets_txt, non_targets_txt):
    with pytest.raises(ValidationError) as error:
        Config(tar_paths=targets_txt, neg_paths=non_targets_txt, stringency=.5, neg_quantile=.5)

    assert {(e['loc'], e['type']) for e in error.value.errors()} == {
        (('stringency',), 'extra_forbidden'),
        (('neg_quantile',), 'extra_forbidden'),
    }


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

    data = json.loads(json_text)
    assert data['version'] == config.version
    assert data['prefix'] == str(tmp_path)
    assert data['tar_paths'] == str(targets_txt)
    assert data['neg_paths'] == str(non_targets_txt)
    assert data['n_tar_th'] is None
    assert data['n_neg_th'] is None
    assert data['neg_rank'] == .2
    assert data['tar_weight'] == .5
    assert data['run_blast'] is False
    assert data['n_cpu'] == 2
    assert data['low_memory'] is True
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == 'test-key'
    assert data['api_key'] == '**********'
    assert 'test-key' not in json_text
