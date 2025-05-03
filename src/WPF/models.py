from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from WPF.data_loader import DataLoader

  
    
class ModelRunner(DataLoader):
    def __init__(self, folder_path, site_index=1, lag_dim=10, forecast_dim=1,
                 train_test_split=0.7, model_type='linear_regression'):
        super().__init__(folder_path, site_index, lag_dim, forecast_dim, train_test_split)
        self.model_type = model_type
        self.Y_pred = None

        if model_type == 'linear_regression':
            self.model = LinearRegression()

        elif model_type == 'svm':
            self.model = SVR()

        elif model_type == 'baseline':
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
    
    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.Y_test, label='True Values')
        plt.plot(self.Y_pred, label='Predicted Values')
        plt.title(f'True vs Predicted ({self.model_type})')
        plt.xlabel('Sample Index')
        plt.ylabel('Power')
        plt.legend()
        plt.show()

    
    def post_process(self):
        df = pd.DataFrame(self.clean_data[self.clean_data.index.isin(self.test_index)]
            ['Power'])
        f = self.forecast_dim
        if f>1:
            for i in range(f):
                column_name = f"Power_{i+1}_hour_predicted"
                s = pd.Series(self.Y_pred[:, i])
                shifted_series = s.shift(i)
                tail = s[-i:] if i > 0 else s[0:0]
                nans = pd.Series(np.repeat(np.nan, f-i-1))
                column_data = pd.concat((shifted_series, tail, nans), ignore_index=True)
                column_data.index = df.index
                df[column_name] = column_data
        else:
            df["Power_predicted"] = self.Y_pred
            
        self.result_df = df


    def execute(self):
        self.train()
        self.predict()
        self.compute_errors()
        self.plot_results()
        

    