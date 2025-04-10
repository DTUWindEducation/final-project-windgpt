import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    
    def __init__(self, file_name, lag_dim, forecast_dim, data_eda_bool):
        
        # Save attributes
        self.raw_data_dir = Path("data") / file_name
        self.lag_dim = lag_dim 
        self.forecast_dim = forecast_dim
        
        # Call methods
        self.data_cleaning()
        self.data_scaling()
        self.data_XY_preparation()
        self.data_processing()
        
        if data_eda_bool:
            self.data_eda() 

    def data_cleaning(self):
        """
        Just a function to read, parse and clean the data
        """
        
        self.raw_data = pd.read_csv(raw_data_dir, index_col=0, parse_dates=True)

        raw_data
        
    def data_scaling(self):
        """function to scale the data
        
        input : cleaned dataset
        output: scaled dataset
        """
        
        
    
    def data_XY_preparation(self):

        """ Here slices for windowing are created
        
        input: cleaned scaled dataset
        output: X Y vectors 
        """
        l = 30
        m = 3
        n = 5

        np_data = raw_data.values   # Convert to the numpy array for slicing data
        N = np_data.shape[0]        # Total number of samples

        k = N - (l + m + n)

        # Create Input and output Slice
        in_slice = np.array([range(i, i + l) for i in range(k)])
        op_slice = np.array([range(i + l + m, i + l + m + n) for i in range(k)])


        in_data = np_data[in_slice,:]
        print(in_data.shape)
        
    def data_eda(self):
        
        """
        function to print some graphs with the Exploratory Data Analysis (EDA)
        """
        
