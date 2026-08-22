import pandas as pd
import pytest
from matplotlib.figure import Figure

from subqcat.viz import SubQCATPlotter


def test_visualisation():
    df = pd.DataFrame({
        "cell_area": ["673", "456"], 
        "some_other_column": [1.0, 2.0], 
        "distance_from_center": [4.0, 7.0], 
        "spatial_cluster": [1, 2]})
    plotter = SubQCATPlotter(df, "cell_area")
    figure = plotter.scatter_plot()
    assert isinstance(figure, Figure)

@pytest.fixture
def missing_metric_dataframe():
    err = pd.DataFrame({
        "some_other_column": [1.0, 2.0], 
        "distance_from_center": [4.0, 7.0], 
        "spatial_cluster": [1, 2]})
    return err

@pytest.fixture
def missing_cols_dataframe():
    err = pd.DataFrame({
        "cell_area": [1.0, 2.0],
        "distance_from_center": [4.0, 7.0], 
        "x_axis": [4.0, 7.0]})
    return err
    
def test_missing_metric_for_vis(missing_metric_dataframe):
    with pytest.raises(ValueError, match="No valid QC columns found for plotting. Expected any of: "):
        plot = SubQCATPlotter(missing_metric_dataframe, 'distance_from_center')
        plot.scatter_plot()

def test_missing_x_metric_for_vis():
    bad_df = pd.DataFrame({"cell_id": [1.0, 2.0], "cell_area": [3.0, 5.0], "distance_from_center": [7.0, 10.0]})
    with pytest.raises(ValueError, match="Specified metric "):
        plot = SubQCATPlotter(bad_df, 'distance_from_center', 'nucleus_area')
        plot.scatter_plot()

def test_metric_not_in_data(missing_metric_dataframe):
    with pytest.raises(ValueError, match="not found in data."):
        plot = SubQCATPlotter(missing_metric_dataframe, "cell_area")
        plot.scatter_plot()

def test_missing_cols_for_vis(missing_cols_dataframe):
    with pytest.raises(ValueError, match="Missing required columns: "):
        plot = SubQCATPlotter(missing_cols_dataframe, "distance_from_center")
        plot.scatter_plot()

