from WPF.data_loader import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

loc1 = DataLoader("inputs/Location2.csv", train_test_pct=0.99)

#loc1.data_eda()

model = LinearRegression()
model.fit(loc1.X_train_2D, loc1.Y_train)
Y_pred = model.predict(loc1.X_test_2D)

print("Mean Squared Error:", mean_squared_error(loc1.Y_test, Y_pred))
print("Mean Absolute Error:", mean_absolute_error(loc1.Y_test, Y_pred))

plt.figure(figsize=(10, 5))
plt.plot(loc1.Y_test, label='True Values')
plt.plot(Y_pred, label='Predicted Values')   
plt.title('True vs Predicted Values')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.show()


