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
from pos_score import pos_score
import gc
import logging

logger = logging.getLogger(__name__)


def _unpack(v):
    return v[0] if isinstance(v, (list, tuple)) else v


###THIS MIGHT BE DEPRECATED WITH POS_COMPARE
#decodes own position using trace and pos from A, then uses it to decide pos from B, compares both to shuffled

NUM_POS_DECODING_REPS = 2

def pos_decoding_AvsB_dep(cell_traceA, posA, cell_traceB, posB, percent_to_train):

    logger.info(f"Starting. cell_traceA: {np.array(cell_traceA).shape}, posA: {np.array(posA).shape}")
    logger.info(f"cell_traceB: {np.array(cell_traceB).shape}, posB: {np.array(posB).shape}, percent_to_train: {percent_to_train}")

    output_dimension = 2 #here, we set as a variable for hypothesis testing below.
    logger.info(f"Initializing CEBRA model (output_dim={output_dimension}, max_iter=8000, lr=5e-6)")
    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= 3e-4,
                            learning_rate= 5.5e-5,
                            #temperature = 2,
                            temperature_mode = 'auto',
                            #min_temperature = .74,
                            output_dimension=output_dimension,
                            max_iterations=8000,
                            #max_iterations=8000,
                            distance='cosine',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 32,
                            time_offsets = 1,
                            #hybrid=True, #added <-- if using time
                            verbose=False)

    logger.info("Initializing shuffle CEBRA model")
    shuff_model =  CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 3e-4,
                        learning_rate= 5.5e-5,
                        #temperature = 2,
                        temperature_mode = 'auto',
                        #min_temperature = .74,
                        output_dimension=output_dimension,
                        max_iterations=8000,
                        #max_iterations=8000,
                        distance='cosine',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 32,
                        time_offsets = 1,
                        #hybrid=True, #added <-- if using time
                        verbose=False)

    ########################
    #TEST



    err_allA = [] * 4
    err_allB_usingA = [] * 4
    err_all_shuffB_usingA = [] * 4
    err_allB_usingB = [] * 4

    logger.info(f"Beginning {NUM_POS_DECODING_REPS} position decoding repetition(s)")
    for i in range(NUM_POS_DECODING_REPS):
        logger.info(f"--- Repetition {i+1}/{NUM_POS_DECODING_REPS} ---")

        logger.info(f"[{i+1}] Splitting envA data ({percent_to_train*100:.0f}/{(1-percent_to_train)*100:.0f} train/test hold-out)")
        cell_trainA, cell_testA = hold_out(cell_traceA, percent_to_train)
        pos_trainA, pos_testA = hold_out(posA, percent_to_train)
        logger.info(f"[{i+1}]   cell_trainA: {np.array(cell_trainA).shape}, cell_testA: {np.array(cell_testA).shape}")

        logger.info(f"[{i+1}] Fitting CEBRA model on envA")
        cebra_loc_modelA = sklearn.base.clone(cebra_loc_model).fit(cell_trainA, pos_trainA) #training on A
        cebra_loc_trainA = cebra_loc_modelA.transform(cell_trainA)
        cebra_loc_testA = cebra_loc_modelA.transform(cell_testA)

        pos_test_scoreA, pos_test_errA, dis_meanA, dis_medianA = pos_score(cebra_loc_trainA, cebra_loc_testA, pos_trainA, pos_testA)
        logger.info(f"[{i+1}] A->A: score={np.mean(pos_test_scoreA):.4f}, err={np.mean(pos_test_errA):.4f}, mean_dist={np.mean(dis_meanA):.4f}")

        cebra_loc_testB = cebra_loc_modelA.transform(cell_traceB)
        pos_test_scoreBwA, pos_test_errBwA, dis_meanBwA, dis_medianBwA = pos_score(cebra_loc_trainA, cebra_loc_testB, pos_trainA, posB)
        logger.info(f"[{i+1}] A->B: score={np.mean(pos_test_scoreBwA):.4f}, err={np.mean(pos_test_errBwA):.4f}, mean_dist={np.mean(dis_meanBwA):.4f}")


        ########################
        #SHUFFLED
        logger.info(f"[{i+1}] Building shuffled control")
        pos_train_shuffA = pos_trainA.copy()
        for column in range(pos_train_shuffA.shape[1]):
            np.random.shuffle(pos_train_shuffA[:, column])

        logger.info(f"[{i+1}] Fitting CEBRA model on shuffled envA position labels")
        shuff_modelA = sklearn.base.clone(cebra_loc_model).fit(cell_trainA, pos_train_shuffA) #training on shuffled A
        cebra_loc_train_shuffA = shuff_modelA.transform(cell_trainA)
        cebra_loc_test_shuffA = shuff_modelA.transform(cell_testA)

        pos_test_score_shuffA, pos_test_err_shuffA, dis_mean_shuffA, dis_median_shuffA = pos_score(cebra_loc_train_shuffA, cebra_loc_test_shuffA, pos_trainA, pos_testA)
        logger.info(f"[{i+1}] Shuff A->A: score={np.mean(pos_test_score_shuffA):.4f}, err={np.mean(pos_test_err_shuffA):.4f}")

        cebra_loc_test_shuffB = shuff_modelA.transform(cell_traceB)
        pos_test_score_shuffB, pos_test_err_shuffB, dis_mean_shuffB, dis_median_shuffB = pos_score(cebra_loc_train_shuffA, cebra_loc_test_shuffB, pos_trainA, posB)
        logger.info(f"[{i+1}] Shuff A->B: score={np.mean(pos_test_score_shuffB):.4f}, err={np.mean(pos_test_err_shuffB):.4f}")


        #then sanity check use B to decode B
        logger.info(f"[{i+1}] Sanity check: fitting envB model (B->B)")
        cell_trainB, cell_testB = hold_out(cell_traceB, percent_to_train)
        pos_trainB, pos_testB = hold_out(posB, percent_to_train)

        cebra_loc_modelB = sklearn.base.clone(cebra_loc_model).fit(cell_trainB, pos_trainB)
        cebra_loc_trainB = cebra_loc_modelB.transform(cell_trainB)
        cebra_loc_testB = cebra_loc_modelB.transform(cell_testB)

        pos_test_scoreB, pos_test_errB, dis_meanB, dis_medianB = pos_score(cebra_loc_trainB, cebra_loc_testB, pos_trainB, pos_testB)
        logger.info(f"[{i+1}] B->B: score={np.mean(pos_test_scoreB):.4f}, err={np.mean(pos_test_errB):.4f}")
        # For err_allA
        err_allA = _unpack(pos_test_scoreA), _unpack(pos_test_errA), _unpack(dis_meanA), _unpack(dis_medianA)

        # For err_allB_usingA
        err_allB_usingA = _unpack(pos_test_scoreBwA), _unpack(pos_test_errBwA), _unpack(dis_meanBwA), _unpack(dis_medianBwA)

        # For err_all_shuffA
        err_all_shuffA = _unpack(pos_test_score_shuffA), _unpack(pos_test_err_shuffA), _unpack(dis_mean_shuffA), _unpack(dis_median_shuffA)

        # For err_all_shuffB_usingA
        err_all_shuffB_usingA = _unpack(pos_test_score_shuffB), _unpack(pos_test_err_shuffB), _unpack(dis_mean_shuffB), _unpack(dis_median_shuffB)

        # For err_allB_usingB
        err_allB_usingB = _unpack(pos_test_scoreB), _unpack(pos_test_errB), _unpack(dis_meanB), _unpack(dis_medianB)


    logger.info(f"Summary | A->A: {np.mean(pos_test_scoreA):.4f}, A->B: {np.mean(pos_test_scoreBwA):.4f}, "
                f"Shuff A->A: {np.mean(pos_test_score_shuffA):.4f}, Shuff A->B: {np.mean(pos_test_score_shuffB):.4f}, "
                f"B->B: {np.mean(pos_test_scoreB):.4f}")

    logger.info("Cleaning up memory")
    del cebra_loc_modelA, cebra_loc_trainA, cebra_loc_testA
    del cebra_loc_testB, shuff_modelA
    del cebra_loc_train_shuffA, cebra_loc_test_shuffA, cebra_loc_test_shuffB
    del cebra_loc_modelB, cebra_loc_trainB

    # Call garbage collector
    gc.collect()
    logger.info("Done. Returning results.")

    return err_allA, err_allB_usingA, err_all_shuffA, err_all_shuffB_usingA, err_allB_usingB
