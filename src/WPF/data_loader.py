"""Importing all necessary modules"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

class DataLoader:
    """
    Load, clean, scale, window, and optionally perform EDA on time-series data.

    Attributes:
        file_path (str): Path to the CSV file.
        lag_dim (int): Number of lag steps for input windows.
        forecast_dim (int): Number of steps to forecast.
        clean_data (pd.DataFrame): Cleaned (dropped columns) data.
        scaled_data (pd.DataFrame): MinMax-scaled data.
        scaler (MinMaxScaler): Fitted scaler instance.
        train_x (np.ndarray): Input feature windows.
        train_y (np.ndarray): Forecast target windows.
    """

    def __init__(self, folder_path, site_index=1, lag_dim=10, forecast_dim=1, train_test_split=0.7,
                 plot_time_series=False, data_eda_bool=False):
        """
        Initialize and run cleaning, scaling, and windowing.

        Args:
            file_path (str): CSV file to load.
            lag_dim (int): Input window length.
            forecast_dim (int): Forecast horizon.
            data_eda_bool (bool): If True, run EDA plots.
        """
        # Save attributes
        self.file_path = folder_path / f"Location{site_index}.csv"
        self.lag_dim = lag_dim
        self.forecast_dim = forecast_dim
        self.train_test_split = train_test_split

        # Call methods
        self.data_cleaning()
        self.data_scaling()
        self.data_xy_preparation()

        if plot_time_series:
            plot_time_series(folder_path,
                             site_index=site_index,
                             variable_name='Power',
                             starting_time='2017-01-01 00:00',
                             ending_time='2021-12-31 23:00')

        if data_eda_bool:
            self.data_eda()

    def data_cleaning(self):
        """
        Read CSV, parse dates, drop unused weather columns.
        """
        # Decide which data columns we want to drop
        df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        self.columns = ['windspeed_100m', 'Power']
        clean_df = df[self.columns]
        self.clean_data = clean_df.copy()
        self.index = clean_df.index

    def data_scaling(self):
        """function to scale the data
        
        input : cleaned dataset
        output: scaled dataset
        """

        # Scale the data between 0 and 1 using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.clean_data[self.columns[:-1]])
        scaled_df = self.clean_data.copy()
        scaled_df[self.columns[:-1]] = scaled_data
        self.scaled_data = scaled_df.copy()

        # Save the scaler for inverse transformation later if needed
        self.scaler = scaler


    def data_xy_preparation(self):

        """ Create sliding windows X (lag_dim) and Y (forecast_dim).
        
        input: cleaned scaled dataset
        output: X Y vectors 
        """
        l = self.lag_dim
        m = self.forecast_dim
        np_data = self.scaled_data.values   # Convert to the numpy array for slicing data
        N = np_data.shape[0]        # Total number of data points; pylint: disable=invalid-name

        # Create empty lists to store the input sequences and output values
        X = []  # pylint: disable=invalid-name
        Y = []  # pylint: disable=invalid-name

        for i in range(N-l-m+1):
            # Create the input sequence (X) and output value (y)
            x = np_data[i:l+i, :]
            y = np_data[l+i:l+m+i, -1]  # Assuming the target variable is the last column

            X.append(x)
            Y.append(y)

        self.X = np.array(X)
        self.X_2D = np.reshape(self.X, (self.X.shape[0], self.X.shape[1]*self.X.shape[2]))
        self.Y = np.array(Y)

        # split the data into training and testing sets
        split_index = int(len(self.Y) * self.train_test_split)
        self.X_train = self.X[:split_index]
        self.X_train_2D = self.X_2D[:split_index]
        self.Y_train = self.Y[:split_index]
        self.X_test = self.X[split_index:]
        self.X_test_2D = self.X_2D[split_index:]
        self.Y_test = self.Y[split_index:]

        self.train_index = self.index[l:self.X_train.shape[0]+l]
        self.test_index = self.index[l+self.X_train.shape[0]:]


    def data_eda(self,
                 variable_name: str = 'Power'):
        """
        Explanatory Data Analysis (EDA):

        Args:
            variable_name (str): Column prefix of the variable to plot 
            (e.g., 'wind_speed_100m' or 'Power').

        Pre-processing:
            - Read the full raw dataset
            - Scale raw data for EDA
        1. Plot raw tine series data of the target variable (Power)
        2. Plot a Histrogramm of the target variable (Power) and kernel density estimates
        3. Plot a boxplot of the target variable grouped by the hour of the day,
            the day of the week, the month of the year
        4. Plot a correlation matrix of all scaled features
        5. Plot autocorrelation and partial autocorrelation plots for the target
            variable (Power)
        6. Plot the seasonal decomposition of the target variable (Power) to identify trends,
            seasonality, and residuals
        """

        df_raw = pd.read_csv(self.file_path, index_col=0, parse_dates=True).sort_index()

        # Construct column name for the selected site
        col = f"{variable_name}"
        if col not in df_raw.columns:
            raise ValueError(f"Column '{col}' not found in data.")

        # Scale raw data for EDA
        scaler_eda = MinMaxScaler()
        df_scaled_raw = pd.DataFrame(
            scaler_eda.fit_transform(df_raw),
            columns=df_raw.columns,
            index=df_raw.index
        )

        # Histograms of scaled raw features
        plt.figure(figsize=(10, 6))
        df_scaled_raw.hist(bins=50)
        plt.suptitle('Histograms of Scaled Raw Features')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Extract time-based features on raw data
        df_raw['hour'] = df_raw.index.hour
        df_raw['day_of_week'] = df_raw.index.day_name()
        df_raw['month'] = df_raw.index.month
        print("Extracted time-based features (first 5 rows):")
        print(df_raw[['hour', 'day_of_week', 'month']].head(), "")

        # 1. Plot raw time series of the selected variable
        plt.figure(figsize=(10, 4))
        plt.plot(df_raw.index, df_raw[col], label=col)
        plt.title(f'{col} over Time')
        plt.xlabel('Time')
        plt.ylabel(variable_name)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2. Histogram + KDE of raw Power
        fig, ax1 = plt.subplots()
        df_raw[col].hist(bins=50, alpha=0.6, ax=ax1, label='Count')
        ax2 = ax1.twinx()
        df_raw[col].plot(kind='kde', ax=ax2, label='KDE')
        ax1.set_xlabel(variable_name)
        ax1.set_ylabel('Count')
        ax2.set_ylabel('Density')
        fig.suptitle(f'{variable_name}: Histogram & KDE')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

        # 3. Boxplots by hour, day, month
        for grp, title in [('hour', 'Hour of Day'),
                           ('day_of_week', 'Day of Week'),
                           ('month', 'Month')]:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df_raw[grp], y=df_raw['Power'])
            plt.title(f'Power by {title}')
            plt.xlabel(title)
            plt.ylabel('Power')
            plt.show()

        # 4. Correlation matrix of scaled raw features
        corr_raw = df_scaled_raw.corr()
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(corr_raw, annot=True, fmt='.2f', cmap='coolwarm',
                         cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix of Scaled Raw Features')
        # Highlight row/column for Power
        idx = corr_raw.columns.get_loc('Power')
        # rectangle around Power column
        ax.add_patch(Rectangle((idx, 0), 1, len(corr_raw), fill=False, edgecolor='black', lw=3))
        # rectangle around Power row
        ax.add_patch(Rectangle((0, idx), len(corr_raw), 1, fill=False, edgecolor='black', lw=3))
        plt.tight_layout()
        plt.show()

        # 5.1 ACF & PACF of raw Power
        plt.figure()
        plot_acf(df_raw['Power'], lags=48)
        plt.title('Autocorrelation of Power')
        plt.show()

        plt.figure()
        plot_pacf(df_raw['Power'], lags=48)
        plt.title('Partial Autocorrelation of Power')
        plt.show()

        # 5.2 Stationarity & ACF/PACF on first-differenced Power
        y = df_raw['Power']
        y_diff = y.diff().dropna()
        plt.figure(figsize=(8,3))
        plot_acf(y_diff, lags=48)
        plt.title('ACF of ΔPower')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8,3))
        plot_pacf(y_diff, lags=48)
        plt.title('PACF of ΔPower')
        plt.tight_layout()
        plt.show()