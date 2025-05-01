from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from WPF.data_loader import DataLoader

class ModelRunner(DataLoader):
    def __init__(self, folder_path, site_index=1, lag_dim=10, forecast_dim=1, train_test_split=0.7, model_type='linear_regression'):
        super().__init__(folder_path, site_index, lag_dim, forecast_dim, train_test_split)
        self.model_type = model_type

        if model_type == 'linear_regression':
            self.model = LinearRegression()


    def train(self):
        self.model.fit(self.X_train_2D, self.Y_train)

    def predict(self):
        self.Y_pred = self.model.predict(self.X_test_2D)
        
    
    def plot_results(self, Y_pred):
        plt.figure(figsize=(10, 5))
        plt.plot(self.Y_test, label='True Values')
        plt.plot(Y_pred, label='Predicted Values')   
        plt.title('True vs Predicted Values')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
    
    def execute(self):
        self.train()
        self.predict()
        self.plot_results(self.Y_pred)
        

    