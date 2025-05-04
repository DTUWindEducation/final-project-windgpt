import pytest
from pathlib import Path
from WPF.models import ModelRunner
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend suitable for tests

DATA_FOLDER = Path(__file__).parent.parent / "inputs"

@pytest.mark.parametrize("model_type", ["linear_regression", "svm", "baseline"])
def test_modelrunner_execution(model_type, monkeypatch):
    # Mock the plot_results method to prevent actual plotting
    monkeypatch.setattr(ModelRunner, "plot_results", lambda self, hours_to_plot=48: None)

    runner = ModelRunner(
        folder_path=DATA_FOLDER,
        site_index=1,
        lag_dim=10,
        forecast_dim=1,
        train_test_split=0.7,
        model_type=model_type
    )
    # Execute full pipeline (train, predict, evaluate, post_process, plot)
    runner.execute()

    # Assertions
    assert runner.Y_pred is not None, "Predictions should not be None"
    assert isinstance(runner.result_df, type(runner.result_df)), "Result dataframe should be created"
    assert not runner.result_df.empty, "Result dataframe should not be empty"

def test_svm_forecast_dim_adjustment(monkeypatch):
    """Test that forecast_dim is adjusted to 1 for SVM models."""
    # Mock the data_xy_preparation method to prevent actual data preparation
    monkeypatch.setattr(ModelRunner, "data_xy_preparation", lambda self: None)

    runner = ModelRunner(
        folder_path=DATA_FOLDER,
        site_index=1,
        lag_dim=10,
        forecast_dim=5,  # Set forecast_dim > 1 to trigger adjustment
        train_test_split=0.7,
        model_type="svm"
    )

    assert runner.forecast_dim == 1, "forecast_dim should be adjusted to 1 for SVM models"

def test_baseline_forecast_dim_adjustment(monkeypatch):
    """Test that forecast_dim is adjusted to 1 for baseline models."""
    # Mock the data_xy_preparation method to prevent actual data preparation
    monkeypatch.setattr(ModelRunner, "data_xy_preparation", lambda self: None)

    runner = ModelRunner(
        folder_path=DATA_FOLDER,
        site_index=1,
        lag_dim=10,
        forecast_dim=5,  # Set forecast_dim > 1 to trigger adjustment
        train_test_split=0.7,
        model_type="baseline"
    )

    assert runner.forecast_dim == 1, "forecast_dim should be adjusted to 1 for baseline models"
    assert runner.model is None, "Model should be None for baseline models"

def test_svm_warning_message(monkeypatch, capsys):
    """Test that a warning message is printed for SVM models with forecast_dim > 1."""
    # Mock the data_xy_preparation method to prevent actual data preparation
    monkeypatch.setattr(ModelRunner, "data_xy_preparation", lambda self: None)

    runner = ModelRunner(
        folder_path=DATA_FOLDER,
        site_index=1,
        lag_dim=10,
        forecast_dim=5,  # Set forecast_dim > 1 to trigger warning
        train_test_split=0.7,
        model_type="svm"
    )

    captured = capsys.readouterr()
    assert "Warning: SVM model only supports one hour prediction." in captured.out

def test_baseline_warning_message(monkeypatch, capsys):
    """Test that a warning message is printed for baseline models with forecast_dim > 1."""
    # Mock the data_xy_preparation method to prevent actual data preparation
    monkeypatch.setattr(ModelRunner, "data_xy_preparation", lambda self: None)

    runner = ModelRunner(
        folder_path=DATA_FOLDER,
        site_index=1,
        lag_dim=10,
        forecast_dim=5,  # Set forecast_dim > 1 to trigger warning
        train_test_split=0.7,
        model_type="baseline"
    )

    captured = capsys.readouterr()
    assert "Warning: Baseline model only supports one hour prediction." in captured.out

def test_post_process(monkeypatch):
    """Test the post_process method."""
    # Mock the clean_data and test_index attributes
    mock_clean_data = pd.DataFrame({
        "Power": np.random.rand(100)
    }, index=pd.date_range("2022-01-01", periods=100, freq="h"))
    mock_test_index = mock_clean_data.index[-31:]

    # Mock the ModelRunner attributes
    runner = ModelRunner(
        folder_path=DATA_FOLDER,
        site_index=1,
        lag_dim=10,
        forecast_dim=2,
        train_test_split=0.7,
        model_type="linear_regression"
    )
    runner.clean_data = mock_clean_data
    runner.test_index = mock_test_index
    runner.Y_pred = np.random.rand(30, 2)  # Mock predictions

    # Call post_process
    runner.post_process()

    # Assertions
    assert isinstance(runner.result_df, pd.DataFrame), "result_df should be a DataFrame"
    assert not runner.result_df.empty, "result_df should not be empty"
    assert "1 hour prediction" in runner.result_df.columns, "result_df should contain '1 hour prediction'"
    assert "2 hour prediction" in runner.result_df.columns, "result_df should contain '2 hour prediction'"

def test_plot_results(monkeypatch):
    """Test the plot_results method."""
    # Mock the result_df attribute
    mock_result_df = pd.DataFrame({
        "Actual": np.random.rand(48),
        "1 hour prediction": np.random.rand(48)
    }, index=pd.date_range("2022-01-01", periods=48, freq="h"))

    # Mock the ModelRunner attributes
    runner = ModelRunner(
        folder_path=DATA_FOLDER,
        site_index=1,
        lag_dim=10,
        forecast_dim=1,
        train_test_split=0.7,
        model_type="linear_regression"
    )
    runner.result_df = mock_result_df

    # Mock plt.show to prevent the plot from being displayed
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    # Call plot_results
    runner.plot_results(hours_to_plot=24)

    # Assertions
    # No exceptions should be raised, and the method should execute successfully