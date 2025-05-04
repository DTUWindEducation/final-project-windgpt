from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from WPF.data_loader import DataLoader

  
    
class ModelRunner(DataLoader):
    def __init__(self, folder_path, site_index: int = 1, lag_dim: int = 10, 
                 forecast_dim: int = 1, train_test_split: float = 0.7, 
                 model_type: str = 'linear_regression'):
        super().__init__(folder_path, site_index, lag_dim, forecast_dim, train_test_split)
        self.model_type = model_type
        self.Y_pred = None

        if model_type == 'linear_regression':
            self.model = LinearRegression()

        elif model_type == 'svm':
            self.model = SVR()
            if self.forecast_dim > 1:
                self.forecast_dim = 1
                self.data_XY_preparation()
                print("Warning: SVM model only supports one hour prediction. Setting forecast_dim to 1.")

        elif model_type == 'baseline':
            if self.forecast_dim > 1:
                print("Warning: Baseline model only supports one hour prediction. Setting forecast_dim to 1.")
                self.forecast_dim = 1  # Set forecast_dim to 1 for persistence model
                self.data_XY_preparation()
            self.model = None  # No training required

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def train(self):
        if self.model_type == 'baseline':
            return  # No training step for persistence model
        self.model.fit(self.X_train_2D, self.Y_train)

    def predict(self):
        if self.model_type == 'baseline':
            # Use the last value of the last feature (assumed to be 'power')
            self.Y_pred = self.X_test[:, -1, -1]
        else:
            self.Y_pred = self.model.predict(self.X_test_2D)
    
    def compute_errors(self):
        mse = mean_squared_error(self.Y_test, self.Y_pred)
        mae = mean_absolute_error(self.Y_test, self.Y_pred)
        r2 = r2_score(self.Y_test, self.Y_pred)

        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R^2 Score:", r2)
    
    def plot_results(self, hours_to_plot=48):
        hours_to_plot = min(hours_to_plot, len(self.result_df.index))
        start_time = self.result_df.index[0]
        end_time = start_time + pd.Timedelta(hours=hours_to_plot)
        sliced_df = self.result_df.loc[start_time:end_time]
        sliced_df.plot(figsize=(12, 6))
        plt.title(f"Model Predictions vs Actual Values ({self.model_type})")
        if hours_to_plot > 48:
            xlabel = f"Time (first {hours_to_plot//24} days of predictions)"
        else:
            xlabel = f"Time (first {hours_to_plot} hours of predictions)"
        plt.xlabel(xlabel)
        plt.ylabel("Power (percentage of max power)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()



    
    def post_process(self):
        df = pd.DataFrame(self.clean_data[self.clean_data.index.isin(self.test_index)]
            ['Power'])
        f = self.forecast_dim
        if f>1:
            for i in range(f):
                column_name = f"{i+1} hour prediction"
                s = pd.Series(self.Y_pred[:, i])
                shifted_series = s.shift(i)
                tail = s[-i:] if i > 0 else s[0:0]
                nans = pd.Series(np.repeat(np.nan, f-i-1))
                column_data = pd.concat((shifted_series, tail, nans), ignore_index=True)
                column_data.index = df.index
                df[column_name] = column_data
        else:
            df["1 hour prediction"] = self.Y_pred
            
        self.result_df = df


    def execute(self):
        self.train()
        self.predict()
        self.compute_errors()
        self.post_process()
        self.plot_results()
        

    