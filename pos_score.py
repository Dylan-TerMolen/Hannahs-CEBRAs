from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
import sklearn.metrics
import sklearn
import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import cebra
from cebra import CEBRA
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection


#scores closeness of position decoding
import sklearn.metrics

# Define decoding function with kNN decoder. For a simple demo, we will use the fixed number of neighbors 36.

def pos_score(emb_train, emb_test, label_train, label_test, n_neighbors=36):
    pos_decoder = cebra.KNNDecoder(n_neighbors, metric = 'euclidean')
    pos_decoder.fit(emb_train, label_train)
    prediction = pos_decoder.predict(emb_test)


    #pos_test_score: The R² score for both position predictions
    #It represents the proportion of variance in the dependent variable that is predictable from the independent variables.
    pos_test_score = sklearn.metrics.r2_score(label_test, prediction)

    #pos_test_err: The median absolute error between the predicted positions and the true positions.
    #This provides a robust measure of the error magnitude.
    pos_test_err = np.median(abs(prediction - label_test))

    # Compute the squared differences for each dimension
    squared_diffs = (prediction - label_test) ** 2
    # Sum the squared differences across columns (axis=1) and take the square root
    distances = np.sqrt(np.sum(squared_diffs, axis=1))

    dis_mean = (np.mean(distances))
    dis_median = (np.median(distances))


    return pos_test_score, pos_test_err, dis_mean, dis_median


def _score_position_subset(prediction, label_test, mask):
    """R^2 and mean Euclidean-distance (cm) error over a boolean subset of a
    test set, or (None, None) if the subset is empty."""
    if not np.any(mask):
        return None, None
    predicted_subset = prediction[mask]
    actual_subset = label_test[mask]
    r2 = sklearn.metrics.r2_score(actual_subset, predicted_subset)
    distances = np.sqrt(np.sum((predicted_subset - actual_subset) ** 2, axis=1))
    cm_error = np.mean(distances)
    return r2, cm_error


def pos_score_by_task_state(emb_train, emb_test, label_train, label_test, in_task_test, n_neighbors=36):
    """Like pos_score, but scores the kNN decoder's test predictions separately
    for in-task and out-of-task rows.

    in_task_test: boolean mask aligned to emb_test/label_test, True where the
    rat was in-task (CS/trace/US trial) at that row.

    Returns {"in_task": (r2, cm_error), "out_of_task": (r2, cm_error)}.
    """
    pos_decoder = cebra.KNNDecoder(n_neighbors, metric='euclidean')
    pos_decoder.fit(emb_train, label_train)
    prediction = pos_decoder.predict(emb_test)

    in_task_test = np.asarray(in_task_test)
    return {
        "in_task": _score_position_subset(prediction, label_test, in_task_test),
        "out_of_task": _score_position_subset(prediction, label_test, ~in_task_test),
    }
