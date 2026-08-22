import pandas as pd
import pytest

from subqcat.clustering import SpatialClusterer
from subqcat.io import XeniumBundle
from subqcat.metrics import SubcellularMetrics
from subqcat.subsample import XeniumSampler


@pytest.fixture
def mock_transcripts_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({
        "cell_id": ["123", "123", "456", "456"], 
        
        "x_location": [10.0, 12.0, 100.0, 120.0], 
        "y_location": [10.0, 10.0, 100.0, 100.0], 
        "z_location": [10.0, 10.0, 100.0, 100.0], 
        
        "qv": [20.0, 20.0, 20.0, 20.0], 
        "is_gene": [True, True, True, True], 
        "codeword_index": [1, 1, 1, 1], 
        "codeword_category": ["gene", "gene", "gene", "gene"]
    })

    fake_data.to_parquet(bundle_dir / "transcripts.parquet")
    bundle = XeniumBundle(bundle_dir)
    return XeniumSampler(bundle)

def test_clustering(mock_transcripts_bundle):
    subsampled_df = mock_transcripts_bundle.subsample()
    metrics = SubcellularMetrics(subsampled_df)
    transcript_distance = metrics.transcript_distance()
    clustering = SpatialClusterer(transcript_distance)
    kmc = clustering.kMeans_clustering('distance_from_center')
    assert isinstance(kmc, pd.DataFrame)
    assert 'spatial_cluster' in kmc.columns

def test_error_clustering():
    bad_df = pd.DataFrame({"some_other_column": [1.0, 2.0]})
    
    with pytest.raises(ValueError, match="Missing required column:"):
        clustering = SpatialClusterer(bad_df)
        clustering.kMeans_clustering('distance_from_center')