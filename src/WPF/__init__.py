import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(folder_path, 
                    site_index,
                    variable_name: str = 'Power',
                    starting_time: str = '2017-01-01 00:00',
                    ending_time: str = '2021-12-31 23:00'):
    """
        1. Plot raw time series data of the target variable (default: Power) for a specified site and time range.
    """
    df_raw = pd.read_csv(folder_path / f"Location{site_index}.csv", index_col=0, parse_dates=True).sort_index()

    df_filtered = df_raw.loc[starting_time:ending_time]
    col = f"{variable_name}"
    if col not in df_filtered.columns:
        raise ValueError(f"Column '{col}' not found in data.")
    plt.figure(figsize=(10, 4))
    plt.plot(df_filtered.index, df_filtered[col], label=col)
    plt.title(f'{col} over Time for Site {site_index}')
    plt.xlabel('Time')
    plt.ylabel(variable_name)
    plt.legend()
    plt.tight_layout()
    plt.show()
