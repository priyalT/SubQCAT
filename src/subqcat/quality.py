import pandas as pd

from subqcat.io import XeniumBundle


class QCEvaluator:
    def __init__(self, clustered_df: pd.DataFrame, bundle: XeniumBundle):
        self.clustered_df = clustered_df
        self.bundle = bundle

    def merge_with_cells(self) -> pd.DataFrame:
        """Merges existing QC with calculate sub-cellular patterns dataset."""
        cells_df = self.bundle.load_cells_dataframe()
        chosen_cells = self.clustered_df['cell_id'].unique()
        filtered_dask_cells = cells_df[cells_df['cell_id'].isin(chosen_cells)].compute()
        final_df = pd.merge(self.clustered_df, filtered_dask_cells, on='cell_id', how='inner')
        return final_df

    def compare_clusters(self, metric: str) -> dict:
        """Compare spatial clusters against known QC metrics."""
        merged_df = self.merge_with_cells()

        qc_columns = ['transcript_counts', 'cell_area', 'nucleus_area']
        available_qc = [col for col in qc_columns if col in merged_df.columns]

        if not available_qc:
            raise ValueError(f"No known QC columns found in cells data. Expected any of: {qc_columns}")
        if metric not in merged_df.columns:
            raise ValueError(f"Metric {metric} not found in merged dataset.")
        grouped_means = merged_df.groupby('spatial_cluster')[[metric] + available_qc].mean()

        correlations = {}
        for col in available_qc:
            correlations[col] = merged_df[metric].corr(merged_df[col])

        return {
            'grouped_means': grouped_means,
            'correlations': correlations,
            'merged_df': merged_df,
        }