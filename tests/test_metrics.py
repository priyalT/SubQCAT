import pandas as pd
import pytest

from subqcat.io import XeniumBundle
from subqcat.metrics import SubcellularMetrics
from subqcat.subsample import XeniumSampler


@pytest.fixture
def mock_transcripts_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({
        "cell_id": ["123", "456"], 
        "x_location": [10.0, 20.0], 
        "y_location": [15.0, 25.0], 
        "z_location": [13.0, 23.0], 
        "qv": [20.0, 20.0], 
        "is_gene": [True, True], 
        "codeword_index": [1, 1], 
        "codeword_category": ["gene", "gene"]
    })
    fake_data.to_parquet(bundle_dir / "transcripts.parquet")
    bundle = XeniumBundle(bundle_dir)
    return XeniumSampler(bundle)

@pytest.fixture
def error_mock_transcripts_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({
        "cell_id": ["123", "456"], 
        "x_location": [10.0, 20.0], 
        "y_location": [15.0, 25.0], 
        "qv": [20.0, 20.0], 
        "is_gene": [True, True], 
        "codeword_index": [1, 1], 
        "codeword_category": ["gene", "gene"]
    })
    fake_data.to_parquet(bundle_dir / "transcripts.parquet")
    bundle = XeniumBundle(bundle_dir)
    return XeniumSampler(bundle)


def test_transcript_distance_calc(mock_transcripts_bundle):
    subsampled_df = mock_transcripts_bundle.subsample()
    metrics = SubcellularMetrics(subsampled_df)
    transcript_distance = metrics.transcript_distance()
    
    assert isinstance(transcript_distance, pd.DataFrame)
    assert 'distance_from_center' in transcript_distance.columns
    assert 'cell_id' in transcript_distance.columns

def test_error_transcript_distance_calc(error_mock_transcripts_bundle):
    subsample_df = error_mock_transcripts_bundle.subsample()
    with pytest.raises(ValueError, match="Missing required columns for metrics calculation:"):
        compute_metrics = SubcellularMetrics(subsample_df)
        compute_metrics.transcript_distance()

