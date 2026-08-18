import dask.dataframe as dd
import numpy as np
import pandas as pd

from subqcat.io import XeniumBundle


class SampleXenium:
    def __init__(self, xenium: XeniumBundle):
        self.xenium = xenium

    def clean_data(self) -> dd.DataFrame:
        df = self.xenium.load_transcripts_dataframe()
        required_cols = ['qv', 'is_gene', 'cell_id', 'codeword_index', 'codeword_category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The dataset is missing required columns for cleaning: {missing_cols}")
        df_quality = df[df['qv']>= 20.0]
        df_gene = df_quality[df_quality['is_gene']==True]
        df_clean = df_gene[df_gene['cell_id'] != "UNASSIGNED"]
        df_clean = df_clean[df_clean['cell_id'] != "-1"]
        df = df_clean.drop(columns=["codeword_index", "codeword_category", "is_gene"])
        return df

    def subsample(self) -> pd.DataFrame:
        clean_df = self.clean_data()
        unique_cells = clean_df['cell_id'].unique().compute()
        if len(unique_cells) == 0:
            raise ValueError("No cells remain after cleaning the data! Cannot subsample.")
        np.random.seed(42)
        sample_size = min(5000, len(unique_cells))
        sampled_cell_ids = np.random.choice(unique_cells, size=sample_size, replace=False)
        df_subsampled = clean_df[clean_df['cell_id'].isin(sampled_cell_ids)]
        df_final = df_subsampled.compute()
        return df_final
