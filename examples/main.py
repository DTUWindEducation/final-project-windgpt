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

#loc1.data_eda()

# model = LinearRegression()
# model.fit(loc1.X_train_2D, loc1.Y_train)
# Y_pred = model.predict(loc1.X_test_2D)

# print("Mean Squared Error:", mean_squared_error(loc1.Y_test, Y_pred))
# print("Mean Absolute Error:", mean_absolute_error(loc1.Y_test, Y_pred))



linear_model = ModelRunner("inputs/Location2.csv", train_test_pct=0.99)

linear_model.execute()
