import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from WPF.data_loader import DataLoader

class ModelRunner(DataLoader):
    """A class to handle the training, prediction, and evaluation of different machine 
    learning models for time series forecasting tasks. Inherits from the DataLoader class 
    to manage data preparation.

    folder_path : str
        Path to the folder containing the dataset.
    site_index : int, optional
        Index of the site to be used for data extraction, by default 1.
    lag_dim : int, optional
        Number of lag features to use for the model, by default 10.
    forecast_dim : int, optional
        Number of steps ahead to forecast, by default 1.
    train_test_split : float, optional
        Proportion of the data to be used for training, by default 0.7.
    model_type : str, optional
        Type of model to use. Supported values are 'linear_regression', 'svm', and 'baseline', 
        by default 'linear_regression'.
    Attributes
    model_type : str
        The type of model being used.
    model : object
        The machine learning model instance (e.g., LinearRegression, SVR, or None for baseline).
    Y_pred : np.ndarray or None
        Predictions made by the model.
    result_df : pd.DataFrame
        DataFrame containing the actual and predicted values for the test set.
    Methods
    -------
    train()
        Trains the model on the training dataset. Does nothing for the baseline model.
    predict()
        Generates predictions using the trained model. For the baseline model, uses persistence.
    compute_errors()
        Computes and prints evaluation metrics (MSE, MAE, R^2) for the predictions.
    plot_results(hours_to_plot=48)
        Plots the actual vs predicted values for a specified number of hours.
    post_process()
        Post-processes the predictions to align them with the test set and prepares the result DataFrame.
    execute()
        Executes the full pipeline: training, prediction, evaluation, post-processing, and plotting.
    Raises
    ------
    ValueError
        If an unsupported model_type is provided.
    """
    def __init__(self, folder_path, site_index: int = 1, lag_dim: int = 10,
                 forecast_dim: int = 1, train_test_split: float = 0.7,
                 model_type: str = 'linear_regression'):
        super().__init__(folder_path, site_index, lag_dim, forecast_dim, train_test_split)
        self.model_type = model_type
        self.Y_pred = None
        self.result_df = None

        if model_type == 'linear_regression':
            self.model = LinearRegression()

        elif model_type == 'svm':
            self.model = SVR()
            if self.forecast_dim > 1:
                self.forecast_dim = 1
                self.data_xy_preparation()
                print("Warning: SVM model only supports one hour prediction. " \
                "Setting forecast_dim to 1.")

        elif model_type == 'baseline':
            if self.forecast_dim > 1:
                print("Warning: Baseline model only supports one hour prediction." \
                " Setting forecast_dim to 1.")
                self.forecast_dim = 1  # Set forecast_dim to 1 for persistence model
                self.data_xy_preparation()
            self.model = None  # No training required

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def train(self):
        """
        Trains the machine learning model using the training dataset.
        
        For non-baseline models, this method fits the model to the prepared
        training data (`X_train_2D` and `Y_train`). For the baseline model,
        no training is required, and the method does nothing.
        """
        if self.model_type == 'baseline':
            return  # No training step for persistence model
        self.model.fit(self.X_train_2D, self.Y_train)

    def predict(self):
        """
        Generates predictions using the trained model.

        For the baseline model, predictions are generated using the persistence
        approach, where the last observed value is used as the prediction.
        For other models, predictions are generated using the model's `predict`
        method on the test dataset (`X_test_2D`).
        """
        if self.model_type == 'baseline':
            # Use the last value of the last feature (assumed to be 'power')
            self.Y_pred = self.X_test[:, -1, -1]
        else:
            self.Y_pred = self.model.predict(self.X_test_2D)

    def compute_errors(self):
        """
        Computes and prints evaluation metrics for the model's predictions.

        The following metrics are calculated:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - R^2 Score

        These metrics are printed to the console for evaluation purposes.
        """
        mse = mean_squared_error(self.Y_test, self.Y_pred)
        mae = mean_absolute_error(self.Y_test, self.Y_pred)
        r2 = r2_score(self.Y_test, self.Y_pred)

        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R^2 Score:", r2)

    def plot_results(self, hours_to_plot=48):
        """
        Plots the actual vs predicted values for a specified number of hours.

        Parameters
        ----------
        hours_to_plot : int, optional
            The number of hours to plot, by default 48. If the specified number
            exceeds the available data, the maximum available data is plotted.

        The plot includes the actual values and the model's predictions, with
        appropriate labels and a grid for better visualization.
        """
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
        """
        Post-processes the predictions to align them with the test set.

        This method prepares a DataFrame (`result_df`) containing the actual
        values and the model's predictions. For multi-step forecasts, the
        predictions are shifted to align with the corresponding time steps.
        """
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
        """
        Executes the full pipeline for the model.

        This method performs the following steps in sequence:
        1. Trains the model (if applicable).
        2. Generates predictions using the trained model.
        3. Computes evaluation metrics for the predictions.
        4. Post-processes the predictions to prepare the result DataFrame.
        5. Plots the actual vs predicted values for visualization.
        """
        self.train()
        self.predict()
        self.compute_errors()
        self.post_process()
        self.plot_results()
