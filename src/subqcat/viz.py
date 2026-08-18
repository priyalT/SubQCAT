import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class SubQCATPlotter:
    def __init__(self, data: pd.DataFrame):
        required_cols = ['cell_area', 'distance_from_center', 'spatial_cluster']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Make sure you pass the merged DataFrame from QualityControl.compare_clusters()."
            )
        self.data = data

    def scatter_plot(self):

        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.data, 
            x='cell_area', 
            y='distance_from_center', 
            hue='spatial_cluster', 
            palette='Set1', 
            s=15,           
            alpha=0.6       
        )

        plt.title("Spatial QC: Cell Area vs. Transcript Spread")
        plt.xlabel("Standard 10x Metric: Cell Area")
        plt.ylabel("Custom Metric: Mean Transcript Distance from Center")

        plt.show()
        return fig
