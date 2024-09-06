import import_ipynb
import pandas as pd
import numpy as np

def simulateMCAR(X, prob_missing):
    matrix_size = X.shape
    np.random.seed(42)
    # Create a random mask to indicate missing values
    missing_mask = np.random.rand(*matrix_size) < prob_missing
    mcar_matrix = np.where(missing_mask, np.nan, X)
    return mcar_matrix

def simulateMNARAllMatrix(X, treshold, prob_missing):
    np.random.seed(42)
    matrix_size = X.shape
    random_values = np.random.rand(*matrix_size)
    missing_mask = np.logical_and(X > treshold, (random_values < prob_missing))
    missing_mask = missing_mask.astype(bool)
    mnar_matrix = np.where(missing_mask, np.nan, X)
    return mnar_matrix

def simulateMNAROneColumn(X, treshold, prob_missing, col):
    np.random.seed(42)
    col_length = X.shape[0]
    random_values = np.random.rand(col_length)
    missing_mask = np.logical_and(X[:, col] > treshold, (random_values < prob_missing))
    missing_mask = missing_mask.astype(bool)
    mnar_matrix = X.copy().astype(float)
    mnar_matrix[missing_mask, col] = np.nan
    return mnar_matrix

def simulateMCAROneColumn(X, prob_missing, col):
    np.random.seed(42)
    col_length = X.shape[0]
    random_values = np.random.rand(col_length)
    missing_mask = random_values < prob_missing
    missing_mask = missing_mask.astype(bool)
    mnar_matrix = X.copy().astype(float)
    mnar_matrix[missing_mask, col] = np.nan
    return mnar_matrix