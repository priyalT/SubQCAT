import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SubQCATPlotter:
    def __init__(self, data: pd.DataFrame, metric: str, x_metric: str | None = None):
        self.data = data
        qc_columns = ['cell_area', 'nucleus_area', 'transcript_counts']
        if metric not in data.columns:
            raise ValueError(f"Metric '{metric}' not found in dataset columns.")
        else:
            self.metric = metric
        if x_metric:
            if x_metric not in data.columns:
                raise ValueError(f"Specified metric '{x_metric}' not found in data.")
            self.x_metric = x_metric
        else:
            available = [col for col in qc_columns if col in data.columns]
            if not available:
                raise ValueError(f"No valid QC columns found for plotting. Expected any of: {qc_columns}")
            self.x_metric = available[0] 

        required_cols = [self.x_metric, self.metric, 'spatial_cluster']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Make sure you pass the merged DataFrame from QualityControl.compare_clusters()."
            )

    def scatter_plot(self):
        fig = plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=self.data, 
            x=self.x_metric, 
            y=self.metric, 
            hue='spatial_cluster', 
            palette='Set1', 
            s=15,           
            alpha=0.6       
        )

        formatted_x_metric = self.x_metric.replace('_', ' ').title()
        formatted_metric = self.metric.replace('_', ' ').title()

        plt.title(f"Spatial QC: {formatted_x_metric} vs. {formatted_metric}")
        plt.xlabel(f"Standard 10x Metric: {formatted_x_metric}")
        plt.ylabel(f"Custom Metric: {formatted_metric}")

        plt.show()
        return fig

