from SubQCAT.io import XeniumBundle
import dask.dataframe as dd
import pandas as pd
import numpy as np

class SampleXenium:
    def __init__(self, xenium: XeniumBundle):
        self.xenium = xenium

    def clean_data(self) -> dd:
        df = self.xenium.load_dataframe()
        df_quality = df[df['qv']>= 20.0]
        df_gene = df_quality[df_quality['is_gene']==True]
        df_clean = df_gene[df_gene['cell_id'] != "UNASSIGNED"]
        df_clean = df_clean[df_clean['cell_id'] != "-1"]
        df = df_clean.drop(columns=["codeword_index", "codeword_category", "is_gene"])
        return df

    def subsample(self) -> pd.DataFrame:
        clean_df = self.clean_data()
        unique_cells = clean_df['cell_id'].unique().compute()
        np.random.seed(42)
        sample_size = min(5000, len(unique_cells))
        sampled_cell_ids = np.random.choice(unique_cells, size=sample_size, replace=False)
        df_subsampled = clean_df[clean_df['cell_id'].isin(sampled_cell_ids)]
        df_final = df_subsampled.compute()
        return df_final
