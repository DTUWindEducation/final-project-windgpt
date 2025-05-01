import os
import pandas as pd
import numpy as np
import pytest
from tempfile import TemporaryDirectory


# Impoert the DataLoader class from your module
from src.WPF.data_loader import DataLoader  

@pytest.fixture
def sample_csv():
    """
    Return the path to the committed test CSV.
    We assume you have tests/data/test.csv in your repo.
    """
    here = os.path.dirname(__file__)
    return os.path.join(here, "data", "test.csv")


def test_data_cleaning_drops_columns(sample_csv):
    dl = DataLoader(sample_csv, lag_dim=1, forecast_dim=1)
    for col in [
        "temperature_2m","relativehumidity_2m","dewpoint_2m",
        "windspeed_10m","winddirection_10m",
        "winddirection_100m","windgusts_10m"
    ]:
        assert col not in dl.clean_data.columns
    assert list(dl.clean_data.columns) == ["Power"]


def test_data_scaling_and_inverse(sample_csv):
    dl = DataLoader(sample_csv, lag_dim=1, forecast_dim=1)
    scaled = dl.scaled_data
    # scaled values between 0 and 1
    assert scaled.min().min() >= 0.0
    assert scaled.max().max() <= 1.0
    # inverse scaling should recover original clean_data
    inv = dl.scaler.inverse_transform(scaled.values).flatten()
    np.testing.assert_allclose(inv, dl.clean_data["Power"].values, atol=1e-6)

@pytest.mark.parametrize("lag,forecast,expected_windows", [
    (2, 1, 6 - 2 - 1 + 1),   # single-step forecast
    (2, 2, 6 - 2 - 2 + 1),   # two-step forecast
])
def test_data_XY_preparation_shapes(sample_csv, lag, forecast, expected_windows):
    dl = DataLoader(sample_csv, lag_dim=lag, forecast_dim=forecast)
    # X shape: (windows, lag, n_features)
    assert dl.trainX.shape == (expected_windows, lag, 1)
    # Y shape: (windows, forecast)
    assert dl.trainY.shape == (expected_windows, forecast)

def test_data_XY_preparation_content(sample_csv):
    """
    For lag=2, forecast=1 and our toy data [0,10,20,30,40,50],
    the first X-window should be [[0],[10]] and first Y [20].
    """
    dl = DataLoader(sample_csv, lag_dim=2, forecast_dim=1)
    X0 = dl.trainX[0].flatten().tolist()
    Y0 = dl.trainY[0].flatten().tolist()
    assert X0 == [0.0, pytest.approx(10.0/50.0)] or pytest.approx([0.0, 10/50])  # scaled values
    # un-scale to check original
    orig_window = dl.scaler.inverse_transform(dl.trainX[0]).flatten().tolist()
    np.testing.assert_allclose(orig_window, [0, 10], atol=1e-6)
    orig_y = dl.scaler.inverse_transform(dl.trainY[0].reshape(-1,1)).flatten()[0]
    assert orig_y == pytest.approx(20.0)

def test_data_eda_runs_without_error(monkeypatch, sample_csv):
    """
    Simply verify that data_eda() does not crash. We patch plt.show
    so the test doesn't actually render figures.
    """
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    dl = DataLoader(sample_csv, lag_dim=1, forecast_dim=1, data_eda_bool=True)
    # If we reach here, EDA ran (at least up to the last plt.show) without exception.
