import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    
    def __init__(self, file_path, lag_dim=10, forecast_dim=1, data_eda_bool=False):
        # TODO: Decide if we want to use file path or folder path as input
        # Save attributes
        self.file_path = file_path
        self.lag_dim = lag_dim 
        self.forecast_dim = forecast_dim
        
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
        N = np_data.shape[0]        # Total number of samples


        trainX = []
        trainY = []

        for i in range(N-l-m+1):
            # Create the input sequence (X) and output value (y)
            X = np_data[i:l+i, :]
            y = np_data[l+i:l+m+i, -1]  # Assuming the target variable is the last column

            trainX.append(X)
            trainY.append(y)

        self.trainX = np.array(trainX)
        self.trainY = np.array(trainY)
        
    def data_eda(self):
        
        """
        function to print some graphs with the Exploratory Data Analysis (EDA)
        """
        
