import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class SpatialClusterer:
    def __init__(self, dataset: pd.DataFrame):
        if 'cell_id' not in dataset.columns:
            raise ValueError("Missing required column: 'cell_id'")
        self.dataset = dataset.copy()

    def kMeans_clustering(self, metric: str, n_clusters: int = 2) -> pd.DataFrame:
        """Cluster cells based on a specified metric using KMeans."""
        if metric not in self.dataset.columns:
            raise ValueError(f"Metric '{metric}' not found in dataset columns.")
        distance_array = np.array(self.dataset[metric]).reshape(-1, 1)
        kmeanModel = KMeans(n_clusters=n_clusters, random_state=42).fit(distance_array)
        self.dataset['spatial_cluster'] = kmeanModel.labels_
        return self.dataset