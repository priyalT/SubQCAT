import dask.dataframe as dd
import pandas as pd
import pytest

from subqcat.io import XeniumBundle
from subqcat.subsample import XeniumSampler


@pytest.fixture
def mock_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({"cell_id": [1, 2], "qv": [30.0, 40.0], "is_gene": [True, True], "codeword_index": [13.0, 23.0], "codeword_category": [1, 2]})
    fake_data.to_parquet(bundle_dir / "transcripts.parquet")
    return XeniumBundle(bundle_dir)

@pytest.fixture
def mock_error_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({"qv": [10.0, 20.0], "is_gene": [True, True], "codeword_index": [13.0, 23.0], "codeword_category": [1, 2]})
    fake_data.to_parquet(bundle_dir / "transcripts.parquet")
    return XeniumBundle(bundle_dir)

@pytest.fixture
def no_unique_cells_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({"cell_id": [1, 1], "qv": [10.0, 20.0], "is_gene": [15.0, 25.0], "codeword_index": [13.0, 23.0], "codeword_category": [1, 2]})
    fake_data.to_parquet(bundle_dir / "transcripts.parquet")
    return XeniumBundle(bundle_dir)

def test_subsampling(mock_bundle):
    sample_xenium = XeniumSampler(mock_bundle)
    final_df = sample_xenium.subsample()
    assert isinstance(final_df, pd.DataFrame)
    assert len(final_df['cell_id'].unique()) <= 5000


def test_cleaning(mock_bundle):
    sample_xenium = XeniumSampler(mock_bundle)
    clean_df = sample_xenium.clean_data()
    assert isinstance(clean_df, dd.DataFrame)
    assert "codeword_index" not in clean_df.columns
    assert "codeword_category" not in clean_df.columns
    assert "is_gene" not in clean_df.columns


def test_cleaning_validation(mock_error_bundle):
    sample_xenium = XeniumSampler(mock_error_bundle)
    with pytest.raises(ValueError):
        sample_xenium.clean_data()

def test_empty_data_after_cleaning(no_unique_cells_bundle):
    sample_xenium = XeniumSampler(no_unique_cells_bundle)
    with pytest.raises(ValueError):
        sample_xenium.subsample()



    


