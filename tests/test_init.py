import pytest
from pathlib import Path
from WPF import plot_time_series

DATA_FOLDER = Path(__file__).parent.parent / "inputs"

def test_plot_time_series(monkeypatch):
    """Test the plot_time_series function with valid inputs."""
    # Mock plt.show to prevent the plot from being displayed
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    # Call the function with valid inputs
    plot_time_series(
        folder_path=DATA_FOLDER,
        site_index=1,
        variable_name="Power",
        starting_time="2017-01-01 00:00",
        ending_time="2017-12-31 23:00"
    )

def test_plot_time_series_invalid_column(monkeypatch):
    """Test the plot_time_series function with an invalid column name."""
    # Mock plt.show to prevent the plot from being displayed
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    with pytest.raises(ValueError, match="Column 'InvalidColumn' not found in data."):
        plot_time_series(
            folder_path=DATA_FOLDER,
            site_index=1,
            variable_name="InvalidColumn",
            starting_time="2017-01-01 00:00",
            ending_time="2017-12-31 23:00"
        )

def test_plot_time_series_invalid_time_range(monkeypatch):
    """Test the plot_time_series function with an invalid time range."""
    # Mock plt.show to prevent the plot from being displayed
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    # Call the function with a time range that has no data
    plot_time_series(
        folder_path=DATA_FOLDER,
        site_index=1,
        variable_name="Power",
        starting_time="1900-01-01 00:00",
        ending_time="1900-12-31 23:00"
    )