import pandas as pd
import pytest

from subqcat.io import XeniumBundle
from subqcat.subsample import SampleXenium


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
    sample_xenium = SampleXenium(mock_bundle)
    sample_xenium.subsample()

def test_cleaning(mock_bundle):
    sample_xenium = SampleXenium(mock_bundle)
    sample_xenium.clean_data()

def test_cleaning_validation(mock_error_bundle):
    sample_xenium = SampleXenium(mock_error_bundle)
    with pytest.raises(ValueError):
        sample_xenium.clean_data()

def test_empty_data_after_cleaning(no_unique_cells_bundle):
    sample_xenium = SampleXenium(no_unique_cells_bundle)
    with pytest.raises(ValueError):
        sample_xenium.subsample()



    


