import pytest
import os
import dask.dataframe as dd
import pandas as pd
from SubQCAT.io import XeniumBundle

@pytest.fixture
def mock_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({"cell_id": [1, 2], "x_location": [10.0, 20.0], "y_location": [15.0, 25.0], "z_location": [13.0, 23.0]})
    fake_data.to_parquet(bundle_dir / "transcripts.parquet")
    return XeniumBundle(bundle_dir)

def test_validation_valid_bundle(mock_bundle):
    mock_bundle.validate()

def test_validation_bundle_does_not_exist(tmp_path):
    bundle = XeniumBundle.validate(tmp_path / "false_data")
    with pytest.raises(FileNotFoundError):
        bundle.validate()

def test_validation_valid_path(tmp_path):
    transcripts = tmp_path / "transcripts.paraquet"
    transcripts.touch()
    bundle = XeniumBundle(transcripts)
    with pytest.raises(ValueError):
        bundle.validate()

def test_validation_valid_missing_transcripts(tmp_path):
    no_transcripts = tmp_path / "empty_bundle"
    bundle = XeniumBundle(no_transcripts)
    with pytest.raises(FileNotFoundError):
        bundle.validate()

def test_valid_transcript_file(mock_bundle):
    assert mock_bundle.transcript_file.is_file()

def test_load_dataframe(mock_bundle):
    df = mock_bundle.load_dataframe()
    assert isinstance(df, dd.DataFrame)

