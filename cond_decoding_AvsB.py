from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
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
from CSUS_score import CSUS_score
from hold_out import hold_out
import gc
import logger



NUM_COND_DECODING_REPS = 2

#decodes conditioning in envB using envA.
#Outputs percent correct in envA after being trained in env A(based on a 75/25 split)
#Outputs percent correct in envB using the model trained in envA


def cond_decoding_AvsB(envA_cell_train, envB_cell_train, envA_eyeblink, envB_eyeblink, dimensions=2):
    dimensions = 2
    logger.info(f"Starting. envA cells: {np.array(envA_cell_train).shape}, envB cells: {np.array(envB_cell_train).shape}")
    logger.info(f"envA eyeblink: {np.array(envA_eyeblink).shape}, envB eyeblink: {np.array(envB_eyeblink).shape}, dimensions: {dimensions}")

    output_dimension = dimensions
    logger.info(f"Initializing CEBRA model (output_dim={output_dimension}, max_iter=8000, lr=8.6e-4)")
#     cebra_loc_model = CEBRA(model_architecture='offset10-model',
#                             batch_size=512,
#                             #learning_rate= .046,
#                             learning_rate= 4.5e-07,
#                             temperature_mode = 'auto',
#                             min_temperature = .25,
#                             output_dimension=3,
#                             max_iterations= 15000, #15000, #<--------------1-20000 # Lowered this from 14000 for testing locally
#                             distance='euclidean',
#                             conditional='time_delta', #added, keep
#                             device='cuda_if_available',
#                             num_hidden_units = 32,
#                             time_offsets = 1,
#                             verbose='true')

    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate= 8.6e-4,
                        temperature_mode = 'auto',
                        min_temperature = .2,
                        output_dimension=output_dimension,
                        max_iterations=8000, 
                        distance='cosine',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 32,
                        time_offsets = 1,
                        verbose=True)

    fract_control_all = []
    fract_test_all = []

    total_fits = NUM_COND_DECODING_REPS * 2
    logger.info(f"Beginning {NUM_COND_DECODING_REPS} conditioning decoding repetitions ({total_fits} CEBRA fits total)")

    for i in range(NUM_COND_DECODING_REPS):
          fit_base = (i * 2) + 1
          logger.info(f"--- Repetition {i+1}/{NUM_COND_DECODING_REPS} ---")

          eyeblink_train_control, eyeblink_test_control = hold_out(envA_eyeblink, .75)
          cell_train_control, cell_test_control = hold_out(envA_cell_train, .75)
          logger.info(f"[{i+1}] cell_train: {np.array(cell_train_control).shape}, cell_test: {np.array(cell_test_control).shape}")

          logger.info(f"[{i+1}] Stage {fit_base}/{total_fits} — Fitting CEBRA on envA (task/eyeblink)")
          cebra_loc_modelpos = sklearn.base.clone(cebra_loc_model).fit(cell_train_control, eyeblink_train_control)
          cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
          cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)

          fract_controlA = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
          logger.info(f"[{i+1}] A->A CSUS score: {fract_controlA:.4f}")

          cell_test = envB_cell_train
          eyeblink_test_control = envB_eyeblink
          cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test)
          fract_testB = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
          logger.info(f"[{i+1}] A->B CSUS score: {fract_testB:.4f}")

          EB_shuff = np.array(envA_eyeblink).copy()
          np.random.shuffle(EB_shuff[:])
          eyeblink_train_control, eyeblink_test_control = hold_out(EB_shuff, .75)
          cell_train_control, cell_test_control = hold_out(envA_cell_train, .75)

          logger.info(f"[{i+1}] Stage {fit_base + 1}/{total_fits} — Fitting CEBRA on envA (shuffled task labels)")
          cebra_loc_modelpos = sklearn.base.clone(cebra_loc_model).fit(cell_train_control, eyeblink_train_control)
          cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
          cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)

          fract_controlA = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
          logger.info(f"[{i+1}] Shuffled A->A CSUS score: {fract_controlA:.4f}")

          fract_control_all.append(fract_controlA)
          fract_test_all.append(fract_testB)
          logger.info(f"[{i+1}] Running totals — control: {fract_control_all}, test: {fract_test_all}")

    logger.info(f"All {NUM_COND_DECODING_REPS} repetitions complete. Cleaning up memory.")
    del cebra_loc_modelpos, cebra_loc_train22, cebra_loc_test22
    gc.collect()

    logger.info(f"Final fract_control_all: {fract_control_all}")
    logger.info(f"Final fract_test_all: {fract_test_all}")

    return fract_control_all, fract_test_all
