[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Wind Forecasting Package

Version 0.0.1

Developed by Marco Saretta, Thomas Røder and Lukas Karkossa in cooperatiom with DTU Wind and Jenni Rinker

## Overview

WPF (Wind Power Forecasting) is a lightweight, modular Python package designed to streamline the end-to-end workflow for wind-power time-series analysis and forecasting. At its basis is a `DataLoader` class that:

1. **Ingests** raw CSV data (power output plus meteorological variables)  
2. **Cleans** it by dropping unneeded columns and parsing timestamps  
3. **Scales** all features into a 0–1 range via MinMax scaling  
4. **Slices** the series into lagged input windows (`X`) and multi‐step targets (`Y`)  
5. (Optionally) **Plots** a battery of EDA graphics: histograms, KDEs, autocorrelations, seasonal decompositions, Welch periodograms, etc.  

Build up on that basis there are three forecasting models integrated: 
1. persistence model
2. linear regression model
3. support vector machine

On top of that foundation you can plug in your own forecasting models—ML or statistical—and easily train, validate, and visualize predictions. WPF requires a few scientific-Python dependencies (NumPy, pandas, scikit-learn, statsmodels, Matplotlib, Seaborn) and can be installed in editable mode for rapid iteration.

## Quick-start guide

How can a user quickly get started with your code?  

In order to quickly run the code like it is, you must fo the following:

**Step 1:** Clone the Repository using git.
git clone <hhttps://github.com/DTUWindEducation/final-project-windgpt>
 - Make sure the working directory is the repository folder

**Step 2:** Create a new environment
 - Create a new environment before installing the required packages. This is not strictly necessary, but it ensures no version conflicts. 

**Step 3:** Required dependencies
- Before installing the WPF, install the required dependencies using pip by typing the below command in an Anaconda prompt or similar (e.g. VS code terminal):
   ``` Anaconda prompt
   pip install numpy==2.2.3 pandas==2.2.3 scikit-learn==1.6.1 statsmodels==0.14.4 matplotlib==3.10.1 seaborn==0.13.2 
   ```
   Note that other version may also work, but the package has been tested with these versions.

- With the dependencies in order, the WPF package can be installed. To install it correclty you need open an Anaconda prompt (or similar) in the "main folder" final-project-windgpt and type the command:
   ```
   pip install .
   ```
   or to install an editable version:
   ```
   pip install -e .
   ```
   
- to check if the package is correctly installed you can use the command pip list and see if the package was installed.


**Step 4:** Inspect the Data:
- Make sure the necessary input files are in the data folder:
    - Location1.csv - Location4.csv
    - if you want to add data for your own location, please make sure it has the same "structure" as the original data. The columns Power and windspeed_100m are mandatory.

**Step 4:** Run the Simulation:
 Execute the main script:
 Go to the examples directory and run the script with
 - python main.py
This script will process the wind data for a specific location and run the selected forecasting models for the specified time period.

**Step 5:** Review Outputs:
- After the simulation completes, review the generated plots displaying the predicted power output for the chosen location.
- The results include visualizations of the model predictions versus actual values, enabling you to assess forecasting
performance.

## Architecture & Functionality

Running the full model is straightforward-simply execute the main.py script. This script orchestrates the following components:
- **ModelRunner Class:** Handles model training, prediction, and evaluation for various forecasting algorithms (e.g., linear regression, SVM, baseline).
- **DataLoader Class:** Loads and preprocesses the input data, performs exploratory data analysis (EDA), and prepares datasets for training and testing.
- **Plotting Functionality:** A utility in __init__.py allows you to visualize raw time series data for any attribute before modeling.
- **Workflow Overview:** The overall workflow is illustrated in the graphical abstract below, showing how data flows from loading, through modeling, to result visualization.
- **Testing:** Additional testing functionalities are provided to verify that the codebase is working as expected.
With this modular structure, you can easily extend or adapt the workflow to new datasets, models, or analysis tasks.

![Workflow Diagram](./architecture.svg)

A description of the code can be also found within the single scripts. An overview of the most important functioanlities is described below.


### ModelRunner Class

#### Overview
The `ModelRunner` class, define in models.py provides a unified interface for training, predicting, evaluating, and visualizing time series forecasting models. It utilizes the `DataLoader` class (described below) for seamless data preparation and supports multiple model types, including linear regression, support vector machines (SVM), and a baseline persistence model.

---


**Parameters**

| Parameter         | Description                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------|
| `folder_path`     | Path to the folder containing the dataset.                                                   |
| `site_index`      | Index of the site to load data from (default: `1`).                                         |
| `lag_dim`         | Number of lag features to use (default: `10`).                                              |
| `forecast_dim`    | Number of steps ahead to forecast (default: `1`).                                           |
| `train_test_split`| Proportion of data for training (default: `0.7`).                                           |
| `model_type`      | Model to use: `'linear_regression'`, `'svm'`, or `'baseline'` (default: `'linear_regression'`). |


---

#### Supported Models

- **Linear Regression**: Standard regression for time series.
- **Support Vector Machine (SVM)**: Regression with support vector machines. Only supports single-step forecasts.
- **Baseline (Persistence)**: Uses the last observed value as the prediction. Only supports single-step forecasts.

---

#### Key Attributes

| Attribute    | Description                                                                              |
|--------------|------------------------------------------------------------------------------------------|
| `model_type` | The type of model being used.                                                            |
| `model`      | The instantiated model object (e.g., `LinearRegression`, `SVR`, or `None` for baseline). |
| `Y_pred`     | Model predictions for the test set.                                                      |
| `result_df`  | DataFrame with actual and predicted values for the test set.                             |

---

#### Core Methods

#### `train()`
Trains the selected model using the training dataset. No training is performed for the baseline model.

#### `predict()`
Generates predictions for the test set:
- For baseline, uses the last observed value (persistence).
- For other models, uses the model's `predict` method.

#### `compute_errors()`
Calculates and prints key regression metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

#### `plot_results(hours_to_plot=48)`
Plots actual vs. predicted values for a specified number of hours (default: 48). Automatically adjusts if less data is available.

#### `post_process()`
Aligns predictions with the test set and prepares the `result_df` DataFrame, handling both single- and multi-step forecasts.

#### `execute()`
Runs the full pipeline:
1. Train the model.
2. Predict on the test set.
3. Compute evaluation metrics.
4. Post-process predictions.
5. Visualize results.

#### Notes

- **Forecast Horizon**: SVM and baseline models only support single-step prediction (`forecast_dim=1`). If a higher value is set, it will be reset to 1 with a warning.
- **Evaluation Metrics**: Uses standard regression metrics (MSE, MAE, R²) for performance assessment.
- **Visualization**: The `plot_results` method provides a quick comparison of actual vs. predicted values for easy interpretation.



### DataLoader

**Purpose**  
Load, clean, scale, window, and (optionally) explore time‐series data from a CSV for downstream forecasting.

**Parameters**  
| Parameter | Description |  
|-----------|-------------|  
| `folder_path` | Path to the folder containing `Location{site_index}.csv` |  
| `site_index` | Site identifier (default: `1`) |  
| `lag_dim` | Look-back window size (default: `10`) |  
| `forecast_dim` | Forecast horizon window size (default: `1`) |  
| `train_test_split` | Fraction of data reserved for training (default: `0.7`) |  
| `plot_time_series` | If `True`, generates a time-series plot (default: `False`) |  
| `data_eda_bool` | If `True`, runs full EDA (default: `False`) |  

---

#### Key Attributes
| Attribute | Description |  
|-----------|-------------|  
| `file_path` | Full path to the CSV file (e.g., `folder_path/Location1.csv`) |  
| `clean_data` | DataFrame after removing unused columns |  
| `scaled_data` | MinMax-scaled DataFrame |  
| `scaler` | Fitted `MinMaxScaler` for inverse transformations |  
| `X`, `Y` | Windowed arrays for features (`n_windows, lag_dim, n_features`) and targets (`n_windows, forecast_dim`) |  
| `X_train`, `Y_train`, `X_test`, `Y_test` | Train/test splits |  
| `train_index`, `test_index` | Timestamps corresponding to each window |  

---

#### Core Methods

##### `data_cleaning()`
- Reads the CSV file and parses dates.
- Retains only `windspeed_100m` and `Power` columns.

##### `data_scaling()`
- Applies MinMax scaling to features.
- Stores scaled data in `scaled_data` and the scaler in `scaler`.

##### `data_xy_preparation()`
- Creates lagged input windows (`X`) and horizon-ahead targets (`Y`).
- Splits data into training and testing sets.
- Records timestamps for each window in `train_index` and `test_index`.

##### `data_eda(variable_name='Power')` *(Optional)*
Generates exploratory plots if `data_eda_bool=True`:
- Histograms and KDE plots
- Boxplots by hour/day/month
- Correlation heatmap
- ACF/PACF plots

---

#### Workflow of data_loader
1. **Initialization**: Automatically runs `data_cleaning()`, `data_scaling()`, and `data_xy_preparation()`.
2. **Optional Plots**: 
   - Time-series visualization if `plot_time_series=True`.
   - Full EDA suite if `data_eda_bool=True`.

---



## Peer review

We embraced a true “four-eyes” development process on every feature:

1. **Pair programming & branch-level review**  
   - Each new feature or bug-fix was first co-authored or stepped through by two developers working together.
2. **Pull-request review by a third teammate**  
   - Once the pair felt confident, a formal GitHub PR was opened and assigned to a third team member for an independent pass:
     - Verified functionality by running tests and example scripts  
     - Checked code style, documentation completeness, and EDA outputs  
     - Left inline comments or suggestions, which were addressed before merging  
3. **Final sanity check**  
   - After merging, one more teammate double-checked the merged main branch: ran linting, re-ran the full test suite, and spot-checked the example plots.

This multi-stage review strategy ensured high code quality, consistent documentation, and robust test coverage across the entire package.
