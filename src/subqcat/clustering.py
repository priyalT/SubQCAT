import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class SpatialClusterer:
    def __init__(self, dataset: pd.DataFrame):
        required_cols = ['cell_id', 'distance_from_center']
        missing_cols = [col for col in required_cols if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for clustering: {missing_cols}")

        self.dataset = dataset.copy()

    def kMeans_clustering(self, n_clusters: int = 2) -> pd.DataFrame:
        """Cluster cells based on their distance_from_center using KMeans."""

        distance_array = np.array(self.dataset['distance_from_center']).reshape(-1, 1)
        kmeanModel = KMeans(n_clusters=n_clusters, random_state=42).fit(distance_array)
        self.dataset['spatial_cluster'] = kmeanModel.labels_
        return self.dataset