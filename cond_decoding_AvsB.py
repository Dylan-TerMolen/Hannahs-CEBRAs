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
from cebra_config import merge_cebra_params
import gc
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)



NUM_COND_DECODING_REPS = 2

# Tuned defaults for cross-environment task/eyeblink decoding. Any subset may be
# overridden per run via the `cebra_params` argument (e.g. from the CLI grid search).
COND_CEBRA_DEFAULTS = dict(
    model_architecture='offset10-model',
    batch_size=512,
    learning_rate=8.6e-4,
    temperature_mode='auto',
    min_temperature=.2,
    output_dimension=2,
    max_iterations=8000,
    distance='cosine',
    conditional='time_delta',
    device='cuda_if_available',
    num_hidden_units=32,
    time_offsets=1,
    verbose=True,
)

#decodes conditioning in envB using envA.
#Outputs percent correct in envA after being trained in env A(based on a 75/25 split)
#Outputs percent correct in envB using the model trained in envA


def cond_decoding_AvsB(envA_cell_train, envB_cell_train, envA_eyeblink, envB_eyeblink, cebra_params=None):
    logger.info(f"Starting. envA cells: {np.array(envA_cell_train).shape}, envB cells: {np.array(envB_cell_train).shape}")
    logger.info(f"envA eyeblink: {np.array(envA_eyeblink).shape}, envB eyeblink: {np.array(envB_eyeblink).shape}")

    params = merge_cebra_params(COND_CEBRA_DEFAULTS, cebra_params)
    logger.info(f"Initializing CEBRA model with params: {params}")
    cebra_loc_model = CEBRA(**params)

    task_a_to_a = []
    task_a_to_b = []
    task_shuffled_a_to_a = []
    task_shuffled_a_to_b = []
    task_b_to_b = []

    total_fits = NUM_COND_DECODING_REPS * 3
    logger.info(f"Beginning {NUM_COND_DECODING_REPS} conditioning decoding repetitions ({total_fits} CEBRA fits total)")

    for i in range(NUM_COND_DECODING_REPS):
          fit_base = (i * 3) + 1
          logger.info(f"--- Repetition {i+1}/{NUM_COND_DECODING_REPS} ---")

          eyeblink_train_control, eyeblink_test_control = hold_out(envA_eyeblink, .75)
          cell_train_control, cell_test_control = hold_out(envA_cell_train, .75)
          logger.info(f"[{i+1}] cell_train: {np.array(cell_train_control).shape}, cell_test: {np.array(cell_test_control).shape}")

          logger.info(f"[{i+1}] Stage {fit_base}/{total_fits} — Fitting CEBRA on envA (task/eyeblink)")
          cebra_loc_modelpos = sklearn.base.clone(cebra_loc_model).fit(cell_train_control, eyeblink_train_control)
          cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
          cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)

          a_to_a = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
          logger.info(f"[{i+1}] A->A CSUS score: {a_to_a:.4f}")

          cebra_loc_testB = cebra_loc_modelpos.transform(envB_cell_train)
          a_to_b = CSUS_score(cebra_loc_train22, cebra_loc_testB, eyeblink_train_control, envB_eyeblink)
          logger.info(f"[{i+1}] A->B CSUS score: {a_to_b:.4f}")

          EB_shuff = np.array(envA_eyeblink).copy()
          np.random.shuffle(EB_shuff[:])
          eyeblink_train_shuff, eyeblink_test_shuff = hold_out(EB_shuff, .75)
          cell_train_shuff, cell_test_shuff = hold_out(envA_cell_train, .75)

          logger.info(f"[{i+1}] Stage {fit_base + 1}/{total_fits} — Fitting CEBRA on envA (shuffled task labels)")
          shuff_model = sklearn.base.clone(cebra_loc_model).fit(cell_train_shuff, eyeblink_train_shuff)
          cebra_loc_train_shuff = shuff_model.transform(cell_train_shuff)
          cebra_loc_test_shuff = shuff_model.transform(cell_test_shuff)

          shuffled_a_to_a = CSUS_score(cebra_loc_train_shuff, cebra_loc_test_shuff, eyeblink_train_shuff, eyeblink_test_shuff)
          logger.info(f"[{i+1}] Shuffled A->A CSUS score: {shuffled_a_to_a:.4f}")

          cebra_loc_test_shuffB = shuff_model.transform(envB_cell_train)
          shuffled_a_to_b = CSUS_score(cebra_loc_train_shuff, cebra_loc_test_shuffB, eyeblink_train_shuff, envB_eyeblink)
          logger.info(f"[{i+1}] Shuffled A->B CSUS score: {shuffled_a_to_b:.4f}")

          eyeblink_trainB, eyeblink_testB = hold_out(envB_eyeblink, .75)
          cell_trainB, cell_testB = hold_out(envB_cell_train, .75)

          logger.info(f"[{i+1}] Stage {fit_base + 2}/{total_fits} — Fitting CEBRA on envB (task/eyeblink)")
          cebra_loc_modelB = sklearn.base.clone(cebra_loc_model).fit(cell_trainB, eyeblink_trainB)
          cebra_loc_trainBB = cebra_loc_modelB.transform(cell_trainB)
          cebra_loc_testBB = cebra_loc_modelB.transform(cell_testB)

          b_to_b = CSUS_score(cebra_loc_trainBB, cebra_loc_testBB, eyeblink_trainB, eyeblink_testB)
          logger.info(f"[{i+1}] B->B CSUS score: {b_to_b:.4f}")

          task_a_to_a.append(a_to_a)
          task_a_to_b.append(a_to_b)
          task_shuffled_a_to_a.append(shuffled_a_to_a)
          task_shuffled_a_to_b.append(shuffled_a_to_b)
          task_b_to_b.append(b_to_b)
          logger.info(f"[{i+1}] Running totals — control: {task_a_to_a}, test: {task_a_to_b}, "
                      f"shuff_control: {task_shuffled_a_to_a}, shuff_test: {task_shuffled_a_to_b}, "
                      f"b_to_b: {task_b_to_b}")

    logger.info(f"All {NUM_COND_DECODING_REPS} repetitions complete. Cleaning up memory.")
    del cebra_loc_modelpos, cebra_loc_train22, cebra_loc_test22, cebra_loc_testB
    del shuff_model, cebra_loc_train_shuff, cebra_loc_test_shuff, cebra_loc_test_shuffB
    del cebra_loc_modelB, cebra_loc_trainBB, cebra_loc_testBB
    gc.collect()

    logger.info(f"Final task_a_to_a: {task_a_to_a}")
    logger.info(f"Final task_a_to_b: {task_a_to_b}")
    logger.info(f"Final task_shuffled_a_to_a: {task_shuffled_a_to_a}")
    logger.info(f"Final task_shuffled_a_to_b: {task_shuffled_a_to_b}")
    logger.info(f"Final task_b_to_b: {task_b_to_b}")

    return task_a_to_a, task_a_to_b, task_shuffled_a_to_a, task_shuffled_a_to_b, task_b_to_b
