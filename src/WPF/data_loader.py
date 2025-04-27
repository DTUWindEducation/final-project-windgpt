import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    
    def __init__(self, file_path, lag_dim=10, forecast_dim=1, train_test_pct=0.8, data_eda_bool=False):
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
        Just a function to read, parse and clean the data
        """
        # TODO: Decide if we want to drop data columns or not
        
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
        self.scaled_data = pd.DataFrame(scaler.fit_transform(self.clean_data), columns=self.clean_data.columns, index=self.clean_data.index)

        # Save the scaler for inverse transformation later if needed
        self.scaler = scaler

        # Save the scaled data to a dataframe
        self.scaled_data = pd.DataFrame(self.scaled_data, columns=self.clean_data.columns, index=self.clean_data.index)
    
    def data_XY_preparation(self):

        """ Here slices for windowing are created
        
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
        split_index = int(len(self.Y) * self.train_test_pct)
        self.X_train = self.X[:split_index]
        self.X_train_2D = self.X_2D[:split_index]
        self.Y_train = self.Y[:split_index]
        self.X_test = self.X[split_index:]
        self.X_test_2D = self.X_2D[split_index:]
        self.Y_test = self.Y[split_index:]
        
    def data_eda(self):
        
        """
        function to print some graphs with the Exploratory Data Analysis (EDA)
        """
        
