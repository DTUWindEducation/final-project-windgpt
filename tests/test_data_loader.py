from pathlib import Path
import pytest
from WPF.data_loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend suitable for tests

# Update this path to the actual folder containing your data
DATA_FOLDER = Path(__file__).parent.parent / "inputs"

def test_data_cleaning():
    """Test the data cleaning process using real data."""
    loader = DataLoader(folder_path=DATA_FOLDER, site_index=1)
    assert "windspeed_100m" in loader.clean_data.columns
    assert "Power" in loader.clean_data.columns
    assert loader.clean_data.isnull().sum().sum() == 0

def test_data_scaling():
    """Test the data scaling process using real data."""
    loader = DataLoader(folder_path=DATA_FOLDER, site_index=1)
    scaled_data = loader.scaled_data
    assert np.isclose(scaled_data["windspeed_100m"].min(), 0, atol=1e-8)
    assert np.isclose(scaled_data["windspeed_100m"].max(), 1.0, atol=1e-8)
def test_data_xy_preparation():
    """Test the sliding window preparation using real data."""
    loader = DataLoader(folder_path=DATA_FOLDER, site_index=1, lag_dim=5, forecast_dim=2)
    assert loader.X.shape[0] == loader.Y.shape[0]
    assert loader.X.shape[1] == 5  # lag_dim
    assert loader.Y.shape[1] == 2  # forecast_dim

def test_train_test_split():
    """Test the train-test split using real data."""
    loader = DataLoader(folder_path=DATA_FOLDER, site_index=1, train_test_split=0.8)
    total_samples = loader.Y.shape[0]
    train_samples = loader.Y_train.shape[0]
    test_samples = loader.Y_test.shape[0]
    assert train_samples == int(total_samples * 0.8)
    assert test_samples == total_samples - train_samples


def test_data_eda(monkeypatch):
    """Test the data_eda method."""
    # Mock plt.show to prevent the plots from being displayed
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    # Initialize DataLoader with EDA enabled
    loader = DataLoader(
        folder_path=DATA_FOLDER,
        site_index=1,
        lag_dim=10,
        forecast_dim=1,
        data_eda_bool=True
    )

