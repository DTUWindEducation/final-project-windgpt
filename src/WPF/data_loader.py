"""Importing all necessary modules"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.signal import welch

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

    def __init__(self, file_path, lag_dim=10, forecast_dim=1, train_test_pct=0.8, data_eda_bool=False):
        """
        Initialize and run cleaning, scaling, and windowing.

        Args:
            file_path (str): CSV file to load.
            lag_dim (int): Input window length.
            forecast_dim (int): Forecast horizon.
            data_eda_bool (bool): If True, run EDA plots.
        """
        # TODO: Decide if we want to use file path or folder path as input
        # Save attributes
        self.file_path = file_path
        self.lag_dim = lag_dim
        self.forecast_dim = forecast_dim
        self.train_test_pct = train_test_pct
        
        # Call methods
        self.data_cleaning()
        self.data_scaling()
        self.data_XY_preparation()
        #self.data_processing()

        if data_eda_bool:
            self.data_eda()

    def data_cleaning(self):
        """
        Read CSV, parse dates, drop unused weather columns.
        """
        # Decide which data columns we want to drop
        df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        clean_df = df.drop(columns=['temperature_2m','relativehumidity_2m','dewpoint_2m',
                                     'windspeed_10m','winddirection_10m',
                                     'winddirection_100m','windgusts_10m'])
        self.clean_data = clean_df.copy()

    def data_scaling(self):
        """function to scale the data
        
        input : cleaned dataset
        output: scaled dataset
        """

        # Scale the data between 0 and 1 using MinMaxScaler
        scaler = MinMaxScaler()
        self.scaled_data = pd.DataFrame(scaler.fit_transform(self.clean_data),
                                        columns=self.clean_data.columns,
                                        index=self.clean_data.index)
        
        # Save the scaler for inverse transformation later if needed
        self.scaler = scaler

        # Save the scaled data to a dataframe
        self.scaled_data = pd.DataFrame(self.scaled_data,
                                        columns=self.clean_data.columns,
                                        index=self.clean_data.index)

    def data_XY_preparation(self):

        """ Create sliding windows X (lag_dim) and Y (forecast_dim).
        
        input: cleaned scaled dataset
        output: X Y vectors 
        """
        l = self.lag_dim
        m = self.forecast_dim
        n = self.scaled_data.shape[1]
        np_data = self.scaled_data.values   # Convert to the numpy array for slicing data
        N = np_data.shape[0]        # Total number of data points

        # Create empty lists to store the input sequences and output values
        X = []
        Y = []

        for i in range(n-l-m+1):
            # Create the input sequence (X) and output value (y)
            x = np_data[i:l+i, :]
            x = np_data[i:l+i, :]
            y = np_data[l+i:l+m+i, -1]  # Assuming the target variable is the last column

            X.append(x)
            Y.append(y)

        self.X = np.array(X)
        self.X_2D = np.reshape(self.X, (self.X.shape[0], self.X.shape[1]*self.X.shape[2]))  
        self.Y = np.array(Y)

        # split the data into training and testing sets
        split_index = int(len(self.Y) * self.train_test_pct)
        self.X_train = self.X[:split_index]
        self.X_train_2D = self.X_2D[:split_index]
        self.Y_train = self.Y[:split_index]
        self.X_test = self.X[split_index:]
        self.X_test_2D = self.X_2D[split_index:]
        self.Y_test = self.Y[split_index:]
        
    def data_eda(self):
        """
        Explanatory Data Analysis (EDA):
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
        7. Plot Welch's method for spectral analysis
        """

        df_raw = pd.read_csv(self.file_path, index_col=0, parse_dates=True).sort_index()

        # Scale raw data for EDA
        scaler_eda = MinMaxScaler()
        df_scaled_raw = pd.DataFrame(
            scaler_eda.fit_transform(df_raw),
            columns=df_raw.columns,
            index=df_raw.index
        )

        # 3. Histograms of scaled raw features
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

        # 1. Plot each year's raw time series of Power in subplots
        years = sorted(df_raw.index.year.unique())
        n_years = len(years)
        fig, axes = plt.subplots(n_years, 1, figsize=(10, 2 * n_years), sharex=False)
        if n_years == 1:
            axes = [axes]
        for ax, year in zip(axes, years):
            df_year = df_raw[df_raw.index.year == year]
            ax.plot(df_year.index, df_year['Power'], label=f'Power {year}')
            ax.set_title(f'Power over Time: {year}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Power')
            ax.legend()
        plt.tight_layout()
        plt.show()

        # 2. Histogram + KDE of raw Power
        fig, ax1 = plt.subplots()
        df_raw['Power'].hist(bins=50, alpha=0.6, ax=ax1, label='Count')
        ax2 = ax1.twinx()
        df_raw['Power'].plot(kind='kde', ax=ax2, color='C1', label='KDE')
        ax1.set_xlabel('Power')
        ax1.set_ylabel('Count')
        ax2.set_ylabel('Density')
        fig.suptitle('Power: Histogram & KDE')
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


        # 6. Multi-seasonal decomposition (weekly & annual)
        try:
            # MSTL now only takes 'periods'
            from statsmodels.tsa.seasonal import MSTL
            decomp = MSTL(y, periods=[168, 8760]).fit()
            fig = decomp.plot()
        except ImportError:
            # first extract weekly seasonality
            stl_week = STL(y, period=168).fit()
            resid = y - stl_week.seasonal
            # then extract annual seasonality from the residual
            stl_year = STL(resid, period=8760).fit()
            fig = stl_year.plot()

        fig.set_size_inches(10, 8)
        plt.tight_layout()
        plt.show()

        # 7. Spectral analysis using Welch's method
        fs = 1.0  # 1 sample per hour
        freqs, psd = welch(y.fillna(method='ffill'), fs=fs,
                        window='hann', nperseg=24*7, noverlap=24*3)
        periods = 1/freqs[1:]
        power_spec = psd[1:]
        plt.figure(figsize=(8,4))
        plt.loglog(periods, power_spec)
        for p,label in [(24,'24 h'), (168,'168 h (weekly)'), (8760,'8760 h (annual)')]:
            plt.axvline(p, linestyle='--', alpha=0.7, label=label)
        plt.xlabel('Period (hours)')
        plt.ylabel('Spectral density')
        plt.title('Welch Periodogram (log–log)')
        plt.legend()
        plt.tight_layout()
        plt.show()
