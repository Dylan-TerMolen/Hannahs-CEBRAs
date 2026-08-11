from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
from hold_out import hold_out
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
from pos_score import pos_score_by_task_state
from cebra_config import merge_cebra_params
import gc
import logging
import sys

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


###THIS MIGHT BE DEPRECATED WITH POS_COMPARE
#decodes own position using trace and pos from A, then uses it to decide pos from B, compares both to shuffled

NUM_POS_DECODING_REPS = 2

# Tuned defaults for cross-environment position decoding. Any subset may be
# overridden per run via the `cebra_params` argument (e.g. from the CLI grid search).
POS_CEBRA_DEFAULTS = dict(
    model_architecture='offset10-model',
    batch_size=512,
    learning_rate=5.5e-5,
    temperature_mode='auto',
    output_dimension=2,
    max_iterations=8000,
    distance='cosine',
    conditional='time_delta',
    device='cuda_if_available',
    num_hidden_units=32,
    time_offsets=1,
    verbose=True,
)


def _format_task_state_scores(scores):
    """Log-friendly summary of a pos_score_by_task_state() result, e.g.
    'in_task r2=0.8100 cm_err=3.2000 | out_of_task r2=0.7900 cm_err=3.5000'."""
    def _format_state(label):
        r2, cm_error = scores[label]
        r2_str = f"{r2:.4f}" if r2 is not None else "n/a"
        cm_str = f"{cm_error:.4f}" if cm_error is not None else "n/a"
        return f"{label} r2={r2_str} cm_err={cm_str}"
    return " | ".join(_format_state(label) for label in ("in_task", "out_of_task"))


def pos_decoding_AvsB_dep(cell_traceA, posA, in_task_A, cell_traceB, posB, in_task_B, percent_to_train, cebra_params=None):
    """in_task_A/in_task_B: boolean masks aligned to cell_traceA/posA and
    cell_traceB/posB, True where the rat was in-task (CS/trace/US trial) at
    that row. Every returned direction is scored separately for in-task vs.
    out-of-task rows -- see pos_score.pos_score_by_task_state."""

    logger.info(f"Starting. cell_traceA: {np.array(cell_traceA).shape}, posA: {np.array(posA).shape}")
    logger.info(f"cell_traceB: {np.array(cell_traceB).shape}, posB: {np.array(posB).shape}, percent_to_train: {percent_to_train}")

    params = merge_cebra_params(POS_CEBRA_DEFAULTS, cebra_params)
    logger.info(f"Initializing CEBRA models with params: {params}")
    cebra_loc_model = CEBRA(**params)
    shuff_model = CEBRA(**params)

    ########################
    #TEST



    place_a_to_a = None
    place_a_to_b = None
    place_shuffled_a_to_a = None
    place_shuffled_a_to_b = None
    place_b_to_b = None

    logger.info(f"Beginning {NUM_POS_DECODING_REPS} position decoding repetition(s)")
    for i in range(NUM_POS_DECODING_REPS):
        logger.info(f"--- Repetition {i+1}/{NUM_POS_DECODING_REPS} ---")

        logger.info(f"[{i+1}] Splitting envA data ({percent_to_train*100:.0f}/{(1-percent_to_train)*100:.0f} train/test hold-out)")
        cell_trainA, cell_testA = hold_out(cell_traceA, percent_to_train)
        pos_trainA, pos_testA = hold_out(posA, percent_to_train)
        _, in_task_testA = hold_out(in_task_A, percent_to_train)
        logger.info(f"[{i+1}]   cell_trainA: {np.array(cell_trainA).shape}, cell_testA: {np.array(cell_testA).shape}")

        fit_base = (i * 3) + 1
        total_fits = NUM_POS_DECODING_REPS * 3
        logger.info(f"[{i+1}] Stage {fit_base}/{total_fits} — Fitting CEBRA on envA (position)")
        cebra_loc_modelA = sklearn.base.clone(cebra_loc_model).fit(cell_trainA, pos_trainA) #training on A
        cebra_loc_trainA = cebra_loc_modelA.transform(cell_trainA)
        cebra_loc_testA = cebra_loc_modelA.transform(cell_testA)

        place_a_to_a = pos_score_by_task_state(cebra_loc_trainA, cebra_loc_testA, pos_trainA, pos_testA, in_task_testA)
        logger.info(f"[{i+1}] A->A: {_format_task_state_scores(place_a_to_a)}")

        cebra_loc_testB = cebra_loc_modelA.transform(cell_traceB)
        place_a_to_b = pos_score_by_task_state(cebra_loc_trainA, cebra_loc_testB, pos_trainA, posB, in_task_B)
        logger.info(f"[{i+1}] A->B: {_format_task_state_scores(place_a_to_b)}")


        ########################
        #SHUFFLED
        logger.info(f"[{i+1}] Building shuffled control")
        pos_train_shuffA = pos_trainA.copy()
        for column in range(pos_train_shuffA.shape[1]):
            np.random.shuffle(pos_train_shuffA[:, column])

        logger.info(f"[{i+1}] Stage {fit_base + 1}/{total_fits} — Fitting CEBRA on envA (shuffled position)")
        shuff_modelA = sklearn.base.clone(cebra_loc_model).fit(cell_trainA, pos_train_shuffA) #training on shuffled A
        cebra_loc_train_shuffA = shuff_modelA.transform(cell_trainA)
        cebra_loc_test_shuffA = shuff_modelA.transform(cell_testA)

        place_shuffled_a_to_a = pos_score_by_task_state(cebra_loc_train_shuffA, cebra_loc_test_shuffA, pos_trainA, pos_testA, in_task_testA)
        logger.info(f"[{i+1}] Shuff A->A: {_format_task_state_scores(place_shuffled_a_to_a)}")

        cebra_loc_test_shuffB = shuff_modelA.transform(cell_traceB)
        place_shuffled_a_to_b = pos_score_by_task_state(cebra_loc_train_shuffA, cebra_loc_test_shuffB, pos_trainA, posB, in_task_B)
        logger.info(f"[{i+1}] Shuff A->B: {_format_task_state_scores(place_shuffled_a_to_b)}")


        #then sanity check use B to decode B
        logger.info(f"[{i+1}] Stage {fit_base + 2}/{total_fits} — Fitting CEBRA on envB (position)")
        cell_trainB, cell_testB = hold_out(cell_traceB, percent_to_train)
        pos_trainB, pos_testB = hold_out(posB, percent_to_train)
        _, in_task_testB = hold_out(in_task_B, percent_to_train)

        cebra_loc_modelB = sklearn.base.clone(cebra_loc_model).fit(cell_trainB, pos_trainB)
        cebra_loc_trainB = cebra_loc_modelB.transform(cell_trainB)
        cebra_loc_testB = cebra_loc_modelB.transform(cell_testB)

        place_b_to_b = pos_score_by_task_state(cebra_loc_trainB, cebra_loc_testB, pos_trainB, pos_testB, in_task_testB)
        logger.info(f"[{i+1}] B->B: {_format_task_state_scores(place_b_to_b)}")

    logger.info(f"Summary | A->A: {_format_task_state_scores(place_a_to_a)}, A->B: {_format_task_state_scores(place_a_to_b)}, "
                f"Shuff A->A: {_format_task_state_scores(place_shuffled_a_to_a)}, Shuff A->B: {_format_task_state_scores(place_shuffled_a_to_b)}, "
                f"B->B: {_format_task_state_scores(place_b_to_b)}")

    logger.info("Cleaning up memory")
    del cebra_loc_modelA, cebra_loc_trainA, cebra_loc_testA
    del cebra_loc_testB, shuff_modelA
    del cebra_loc_train_shuffA, cebra_loc_test_shuffA, cebra_loc_test_shuffB
    del cebra_loc_modelB, cebra_loc_trainB

    # Call garbage collector
    gc.collect()
    logger.info("Done. Returning results.")

    return place_a_to_a, place_a_to_b, place_shuffled_a_to_a, place_shuffled_a_to_b, place_b_to_b
