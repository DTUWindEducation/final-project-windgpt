import numpy as np
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def scale_data(data):
    return None


def train_test_split(number_hours_input, number_hours_output, data):
    return None


def fit_model(model, Xtrain, ytrain):
    return None


def cross_validation(model, Xtrain, ytrain):
    return None


def predict(model, Xtest):
    return None

