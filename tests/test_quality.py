import pandas as pd
import pytest

from subqcat.clustering import Cluster
from subqcat.io import XeniumBundle
from subqcat.quality import QualityControl
from subqcat.subsample import SampleXenium


@pytest.fixture
def mock_transcripts_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_transcripts_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({
        "cell_id": ["123", "123", "456", "456"], 
        
        "x_location": [10.0, 12.0, 100.0, 120.0], 
        "y_location": [10.0, 10.0, 100.0, 100.0], 
        "z_location": [10.0, 10.0, 100.0, 100.0], 
        "distance_from_center": [10.0, 20.0, 30.0, 40.0],
        "qv": [20.0, 20.0, 20.0, 20.0], 
        "is_gene": [True, True, True, True], 
        "spatial_cluster": [1, 1, 1, 1], 
        "codeword_category": ["gene", "gene", "gene", "gene"],
        "codeword_index": [1, 2, 3, 4]
    })

    fake_data.to_parquet(bundle_dir / "transcripts.parquet")
    bundle = XeniumBundle(bundle_dir)
    sub_sample = SampleXenium(bundle)
    sub_sample = sub_sample.subsample()
    clustering = Cluster(sub_sample)
    return clustering.kMeans_clustering()


@pytest.fixture
def mock_cells_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_cells_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({
            "cell_id": ["123", "456"], 
            "transcript_counts": [10.0, 100.0], 
            "cell_area": [10.0, 100.0], 
            "nucleus_area": [10.0, 100.0],     
        })
    fake_data.to_parquet(bundle_dir / "cells.parquet")
    bundle = XeniumBundle(bundle_dir)
    return bundle

@pytest.fixture
def mock_err_bundle(tmp_path):
    bundle_dir = tmp_path / "mock_cells_xenium_bundle"
    bundle_dir.mkdir()
    fake_data = pd.DataFrame({
            "cell_id": ["123", "456"]
        })
    fake_data.to_parquet(bundle_dir / "cells.parquet")
    bundle = XeniumBundle(bundle_dir)
    return bundle


def test_cell_merging(mock_transcripts_bundle, mock_cells_bundle):
    qc = QualityControl(mock_transcripts_bundle, mock_cells_bundle)
    merged_cells = qc.merge_with_cells()
    assert isinstance(merged_cells, pd.DataFrame)

def test_cell_qc(mock_transcripts_bundle, mock_cells_bundle):
    qc = QualityControl(mock_transcripts_bundle, mock_cells_bundle)
    qc_dict = qc.compare_clusters()
    assert isinstance(qc_dict, dict)

def test_error_cellqc(mock_transcripts_bundle, mock_err_bundle):
    qc = QualityControl(mock_transcripts_bundle, mock_err_bundle)
    with pytest.raises(ValueError, match="No known QC columns found in cells data. Expected any of:"):
        qc.compare_clusters()