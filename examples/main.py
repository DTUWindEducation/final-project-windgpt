from pathlib import Path
from WPF import plot_time_series
from WPF.data_loader import DataLoader
from WPF.models import ModelRunner


folder_path = Path(__file__).parent.parent / "inputs"

# plot any column in the data set as a time series
plot_time_series(
                 folder_path=folder_path,
                 site_index=1,
                 variable_name= 'Power',
                 starting_time = '2018-01-01 00:00',
                 ending_time = '2019-06-30 23:00')

# If desired, you can also run the data loader to create a DataLoader object
# This is useful for analysing the data by running the exploratory data analysis (EDA), 
# in order to see the data distribution and correlation between the variables.
# It also prepares the data for training and testing by creating the X and Y datasets.
# It is not necessary to run on its own, since the ModelRunner class inherits 
# from the DataLoader class.
loc1 = DataLoader(folder_path,
                  site_index=1,
                  lag_dim=10,
                  forecast_dim=3,
                  train_test_split=0.70,
                  data_eda_bool=True)

# Call the ModelRunner class to run the models
# The liner regression model is able to forecast multiple hours ahead, here set to 3 hours
linear_runner = ModelRunner(folder_path, 
                            site_index=1,
                            lag_dim=10,
                            forecast_dim=3, 
                            train_test_split=0.70,
                            model_type='linear_regression',)
linear_runner.execute()

# The SVM model is only able to forecast one hour ahead, so the forecast_dim will be overwritten
# to 1 hour even if it is set to something else
svm_runner = ModelRunner(folder_path, model_type='svm', forecast_dim=3)
svm_runner.execute()

# The baseline model is a persistence model, which means it uses the last known value. 
# It also only forecasts one hour ahead, so the forecast_dim will be overwritten to 1 hour
baseline_runner = ModelRunner(folder_path, model_type='baseline')
baseline_runner.execute()


# The ModelRunner class will automatically plot the first 48 hours of the 
# results of the model predictions. This can be changed by calling the
# plot_results method and changing the hours_to_plot parameter, e g a week:
linear_runner.plot_results(hours_to_plot=7*24)
