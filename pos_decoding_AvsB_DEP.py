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

###THIS MIGHT BE DEPRECATED WITH POS_COMPARE
#decodes own position using trace and pos from A, then uses it to decide pos from B, compares both to shuffled

NUM_POS_DECODING_REPS = 2

def pos_decoding_AvsB_dep(cell_traceA, posA, cell_traceB, posB, percent_to_train):

    print(f"[pos_decoding_AvsB_dep] Starting. cell_traceA: {np.array(cell_traceA).shape}, posA: {np.array(posA).shape}")
    print(f"[pos_decoding_AvsB_dep] cell_traceB: {np.array(cell_traceB).shape}, posB: {np.array(posB).shape}, percent_to_train: {percent_to_train}")

    output_dimension = 2 #here, we set as a variable for hypothesis testing below.
    print(f"[pos_decoding_AvsB_dep] Initializing CEBRA model (output_dim={output_dimension}, max_iter=8000, lr=5e-6)")
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

    print(f"[pos_decoding_AvsB_dep] Initializing shuffle CEBRA model")
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

    print(f"[pos_decoding_AvsB_dep] Beginning {NUM_POS_DECODING_REPS} position decoding repetition(s)")
    for i in range(NUM_POS_DECODING_REPS):
        print(f"[pos_decoding_AvsB_dep] --- Repetition {i+1}/{NUM_POS_DECODING_REPS} ---")

        print(f"[pos_decoding_AvsB_dep] [{i+1}] Splitting envA data ({percent_to_train*100:.0f}/{(1-percent_to_train)*100:.0f} train/test hold-out)")
        cell_trainA, cell_testA = hold_out(cell_traceA, percent_to_train)
        pos_trainA, pos_testA = hold_out(posA, percent_to_train)
        print(f"[pos_decoding_AvsB_dep] [{i+1}]   cell_trainA: {np.array(cell_trainA).shape}, cell_testA: {np.array(cell_testA).shape}")

        print(f"[pos_decoding_AvsB_dep] [{i+1}] Fitting CEBRA model on envA (cell_trainA + pos_trainA)")
        cebra_loc_modelA = sklearn.base.clone(cebra_loc_model).fit(cell_trainA, pos_trainA) #training on A
        print(f"[pos_decoding_AvsB_dep] [{i+1}] Transforming envA train and test splits")
        cebra_loc_trainA = cebra_loc_modelA.transform(cell_trainA) #training on A
        cebra_loc_testA = cebra_loc_modelA.transform(cell_testA) #testing on A

        print(f"[pos_decoding_AvsB_dep] [{i+1}] Scoring envA position decoding (A->A)")
        pos_test_scoreA, pos_test_errA, dis_meanA, dis_medianA = pos_score(cebra_loc_trainA, cebra_loc_testA, pos_trainA, pos_testA)
        print(f"[pos_decoding_AvsB_dep] [{i+1}]   pos_score A->A: {np.mean(pos_test_scoreA):.4f}, err: {np.mean(pos_test_errA):.4f}, mean_dist: {np.mean(dis_meanA):.4f}")
        #want pos_test_err

        print(f"[pos_decoding_AvsB_dep] [{i+1}] Applying envA model to envB (cross-environment decoding)")
        cebra_loc_testB = cebra_loc_modelA.transform(cell_traceB) #training on A, testing on A
        pos_test_scoreBwA, pos_test_errBwA, dis_meanBwA, dis_medianBwA = pos_score(cebra_loc_trainA, cebra_loc_testB, pos_trainA, posB)
        print(f"[pos_decoding_AvsB_dep] [{i+1}]   pos_score A->B: {np.mean(pos_test_scoreBwA):.4f}, err: {np.mean(pos_test_errBwA):.4f}, mean_dist: {np.mean(dis_meanBwA):.4f}")


        ########################
        #SHUFFLED
        print(f"[pos_decoding_AvsB_dep] [{i+1}] Building shuffled control: shuffling pos_trainA columns independently")
        # Create a new array to hold the shuffled data
        pos_train_shuffA = pos_trainA.copy()
        # Shuffle each column independently
        for column in range(pos_train_shuffA.shape[1]):
            np.random.shuffle(pos_train_shuffA[:, column])

        # Fit the model with the shuffled data
        print(f"[pos_decoding_AvsB_dep] [{i+1}] Fitting CEBRA model on shuffled envA position labels")
        shuff_modelA = sklearn.base.clone(cebra_loc_model).fit(cell_trainA, pos_train_shuffA) #training on shuffled A
        print(f"[pos_decoding_AvsB_dep] [{i+1}] Transforming with shuffled model (envA train and test)")
        cebra_loc_train_shuffA = shuff_modelA.transform(cell_trainA) #training on A
        cebra_loc_test_shuffA = shuff_modelA.transform(cell_testA) #testing on A

        pos_test_score_shuffA, pos_test_err_shuffA, dis_mean_shuffA, dis_median_shuffA = pos_score(cebra_loc_train_shuffA, cebra_loc_test_shuffA, pos_trainA, pos_testA)
        print(f"[pos_decoding_AvsB_dep] [{i+1}]   pos_score shuffled A->A: {np.mean(pos_test_score_shuffA):.4f}, err: {np.mean(pos_test_err_shuffA):.4f}")

        print(f"[pos_decoding_AvsB_dep] [{i+1}] Applying shuffled envA model to envB")
        cebra_loc_test_shuffB = shuff_modelA.transform(cell_traceB) #testing on A
        pos_test_score_shuffB, pos_test_err_shuffB, dis_mean_shuffB, dis_median_shuffB = pos_score(cebra_loc_train_shuffA, cebra_loc_test_shuffB, pos_trainA, posB)
        print(f"[pos_decoding_AvsB_dep] [{i+1}]   pos_score shuffled A->B: {np.mean(pos_test_score_shuffB):.4f}, err: {np.mean(pos_test_err_shuffB):.4f}")


        #then sanity check use B to decode B
        print(f"[pos_decoding_AvsB_dep] [{i+1}] Sanity check: fitting envB model to decode envB (B->B)")
        cell_trainB, cell_testB = hold_out(cell_traceB, percent_to_train)
        pos_trainB, pos_testB = hold_out(posB, percent_to_train)
        print(f"[pos_decoding_AvsB_dep] [{i+1}]   cell_trainB: {np.array(cell_trainB).shape}, cell_testB: {np.array(cell_testB).shape}")

        cebra_loc_modelB = sklearn.base.clone(cebra_loc_model).fit(cell_trainB, pos_trainB)
        print(f"[pos_decoding_AvsB_dep] [{i+1}] Transforming envB train and test splits")
        cebra_loc_trainB = cebra_loc_modelB.transform(cell_trainB)
        cebra_loc_testB = cebra_loc_modelB.transform(cell_testB)

        pos_test_scoreB, pos_test_errB, dis_meanB, dis_medianB = pos_score(cebra_loc_trainB, cebra_loc_testB, pos_trainB, pos_testB)
        print(f"[pos_decoding_AvsB_dep] [{i+1}]   pos_score B->B: {np.mean(pos_test_scoreB):.4f}, err: {np.mean(pos_test_errB):.4f}")
        #want pos_test_err


        print(f"[pos_decoding_AvsB_dep] [{i+1}] Packaging results into output tuples")
        # For err_allA
        pos_test_scoreA_val = pos_test_scoreA[0] if isinstance(pos_test_scoreA, (list, tuple)) else pos_test_scoreA
        pos_test_errA_val = pos_test_errA[0] if isinstance(pos_test_errA, (list, tuple)) else pos_test_errA
        dis_meanA_val = dis_meanA[0] if isinstance(dis_meanA, (list, tuple)) else dis_meanA
        dis_medianA_val = dis_medianA[0] if isinstance(dis_medianA, (list, tuple)) else dis_medianA

        # Create the tuple
        err_allA = pos_test_scoreA_val, pos_test_errA_val, dis_meanA_val, dis_medianA_val

        # For err_allB_usingA
        pos_test_scoreBwA_val = pos_test_scoreBwA[0] if isinstance(pos_test_scoreBwA, (list, tuple)) else pos_test_scoreBwA
        pos_test_errBwA_val = pos_test_errBwA[0] if isinstance(pos_test_errBwA, (list, tuple)) else pos_test_errBwA
        dis_meanBwA_val = dis_meanBwA[0] if isinstance(dis_meanBwA, (list, tuple)) else dis_meanBwA
        dis_medianBwA_val = dis_medianBwA[0] if isinstance(dis_medianBwA, (list, tuple)) else dis_medianBwA

        # Create the tuple
        err_allB_usingA = pos_test_scoreBwA_val, pos_test_errBwA_val, dis_meanBwA_val, dis_medianBwA_val

        # For err_all_shuffA
        pos_test_score_shuffA_val = pos_test_score_shuffA[0] if isinstance(pos_test_score_shuffA, (list, tuple)) else pos_test_score_shuffA
        pos_test_err_shuffA_val = pos_test_err_shuffA[0] if isinstance(pos_test_err_shuffA, (list, tuple)) else pos_test_err_shuffA
        dis_mean_shuffA_val = dis_mean_shuffA[0] if isinstance(dis_mean_shuffA, (list, tuple)) else dis_mean_shuffA
        dis_median_shuffA_val = dis_median_shuffA[0] if isinstance(dis_median_shuffA, (list, tuple)) else dis_median_shuffA

        # Create the tuple
        err_all_shuffA = pos_test_score_shuffA_val, pos_test_err_shuffA_val, dis_mean_shuffA_val, dis_median_shuffA_val

        # For err_all_shuffB_usingA
        pos_test_score_shuffB_val = pos_test_score_shuffB[0] if isinstance(pos_test_score_shuffB, (list, tuple)) else pos_test_score_shuffB
        pos_test_err_shuffB_val = pos_test_err_shuffB[0] if isinstance(pos_test_err_shuffB, (list, tuple)) else pos_test_err_shuffB
        dis_mean_shuffB_val = dis_mean_shuffB[0] if isinstance(dis_mean_shuffB, (list, tuple)) else dis_mean_shuffB
        dis_median_shuffB_val = dis_median_shuffB[0] if isinstance(dis_median_shuffB, (list, tuple)) else dis_median_shuffB

        # Create the tuple
        err_all_shuffB_usingA = pos_test_score_shuffB_val, pos_test_err_shuffB_val, dis_mean_shuffB_val, dis_median_shuffB_val

        # For err_allB_usingB
        pos_test_scoreB_val = pos_test_scoreB[0] if isinstance(pos_test_scoreB, (list, tuple)) else pos_test_scoreB
        pos_test_errB_val = pos_test_errB[0] if isinstance(pos_test_errB, (list, tuple)) else pos_test_errB
        dis_meanB_val = dis_meanB[0] if isinstance(dis_meanB, (list, tuple)) else dis_meanB
        dis_medianB_val = dis_medianB[0] if isinstance(dis_medianB, (list, tuple)) else dis_medianB

        # Create the tuple
        err_allB_usingB = pos_test_scoreB_val, pos_test_errB_val, dis_meanB_val, dis_medianB_val


    print(f"[pos_decoding_AvsB_dep] Summary of mean pos scores:")
    print(f"[pos_decoding_AvsB_dep]   A->A:          {np.mean(pos_test_scoreA):.4f}")
    print(f"[pos_decoding_AvsB_dep]   A->B:          {np.mean(pos_test_scoreBwA):.4f}")
    print(f"[pos_decoding_AvsB_dep]   Shuffled A->A: {np.mean(pos_test_score_shuffA):.4f}")
    print(f"[pos_decoding_AvsB_dep]   Shuffled A->B: {np.mean(pos_test_score_shuffB):.4f}")
    print(f"[pos_decoding_AvsB_dep]   B->B:          {np.mean(pos_test_scoreB):.4f}")
    print(np.mean(pos_test_scoreA))
    print(np.mean(pos_test_scoreBwA))
    print(np.mean(pos_test_score_shuffA))
    print(np.mean(pos_test_score_shuffB))
    print(np.mean(pos_test_scoreB))

    print(f"[pos_decoding_AvsB_dep] Cleaning up memory")
    del cebra_loc_modelA, cebra_loc_trainA, cebra_loc_testA
    del cebra_loc_testB, shuff_modelA
    del cebra_loc_train_shuffA, cebra_loc_test_shuffA, cebra_loc_test_shuffB
    del cebra_loc_modelB, cebra_loc_trainB

    # Call garbage collector
    gc.collect()
    print(f"[pos_decoding_AvsB_dep] Done. Returning results.")

    return err_allA, err_allB_usingA, err_all_shuffA, err_all_shuffB_usingA, err_allB_usingB
