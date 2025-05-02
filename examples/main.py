from pathlib import Path
from WPF.data_loader import DataLoader
from WPF.models import ModelRunner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

folder_path = Path(__file__).parent.parent / "inputs"

loc1 = DataLoader(folder_path, train_test_split=0.70, site_index=1)

# loc1.plot_time_series(
#     folder_path=folder_path,
#     site_index=1,
#     variable_name= 'Power',
#     starting_time = '2017-07-01 00:00',
#     ending_time = '2019-06-30 23:00')

# loc1.data_eda()

# linear_runner = ModelRunner(folder_path, model_type='linear_regression', train_test_split=0.70)
# linear_runner.execute()

# svm_runner = ModelRunner(folder_path, model_type='svm', train_test_split=0.70)
# svm_runner.execute()

# baseline_runner = ModelRunner(folder_path, model_type='baseline', train_test_split=0.70)
# baseline_runner.execute()

m  = ModelRunner(folder_path, forecast_dim=3)
m.execute()
m.post_process()

n = ModelRunner(folder_path, forecast_dim=1)