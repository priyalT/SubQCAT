import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial import Delaunay



class SubcellularMetrics:
    def __init__(self, dataset: pd.DataFrame):
        required_cols = ['cell_id', 'x_location', 'y_location', 'z_location']
        missing_cols = [col for col in required_cols if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for metrics calculation: {missing_cols}")
        self.dataset = dataset.copy()

    def transcript_distance(self) -> pd.DataFrame:
        """Calculate the mean transcript distance to cell centroid for each cell."""
        df_grouped_by_cell = self.dataset.groupby('cell_id')
        centroids = df_grouped_by_cell[['x_location', 'y_location', 'z_location']].transform('mean')
        self.dataset['centroid_x'] = centroids['x_location']
        self.dataset['centroid_y'] = centroids['y_location']
        self.dataset['centroid_z'] = centroids['z_location']
        self.dataset['distance_from_center'] = np.sqrt(
            (self.dataset['x_location'] - self.dataset['centroid_x'])**2 +
            (self.dataset['y_location'] - self.dataset['centroid_y'])**2 +
            (self.dataset['z_location'] - self.dataset['centroid_z'])**2
        )
        mean_distance = self.dataset.groupby('cell_id')[['distance_from_center']].mean()
        return mean_distance.reset_index()

    def nearest_neighbour_transcript(self) -> pd.DataFrame:
        """Calculate the mean nearest neighbour distance between transcripts for each cell."""
        df_grouped_by_cell = self.dataset.groupby('cell_id')
        nn_distances = []
        for cell_id, group in df_grouped_by_cell:
            if len(group) < 2:
                nn_distances.append({'cell_id' : cell_id, 'mean_nn_distance': np.nan})
                continue
            coords = group[['x_location', 'y_location', 'z_location']].values
            tree = KDTree(coords)
            distances, _ = tree.query(coords, k=2)
            distances = np.asarray(distances)
            actual_nn_distances = distances[:, 1]
            mean_dist = np.mean(actual_nn_distances)
            nn_distances.append({'cell_id': cell_id, 'mean_nn_distance': mean_dist})
        return pd.DataFrame(nn_distances)

    def triangulation(self) -> pd.DataFrame:
        """Calculate the Delaunay triangulation graphs for each cell."""
        df_grouped_by_cell = self.dataset.groupby('cell_id')
        triangulation = []
        for cell_id, group in df_grouped_by_cell:
            if len(group) < 4:
                triangulation.append({'cell_id': cell_id, 'edge_length_variance': np.nan})
                continue
            coords = group[['x_location', 'y_location']].values
            tri = Delaunay(coords)
            edges = set()
            for simplex in tri.simplices:  
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                    edges.add(edge)
            
            edge_lengths = []
            for i, j in edges:
                length = np.linalg.norm(coords[i] - coords[j])
                edge_lengths.append(length)
            
            triangulation.append({
                'cell_id': cell_id, 
                'edge_length_variance': np.var(edge_lengths)
            })
        results = pd.DataFrame(triangulation)
        results['log_variance'] = np.log1p(results['edge_length_variance'])
