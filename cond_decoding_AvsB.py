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




NUM_COND_DECODING_REPS = 2

#decodes conditioning in envB using envA.
#Outputs percent correct in envA after being trained in env A(based on a 75/25 split)
#Outputs percent correct in envB using the model trained in envA


def cond_decoding_AvsB(envA_cell_train, envB_cell_train, envA_eyeblink, envB_eyeblink, dimensions=2):

    print(f"[cond_decoding_AvsB] Starting. envA cells: {np.array(envA_cell_train).shape}, envB cells: {np.array(envB_cell_train).shape}")
    print(f"[cond_decoding_AvsB] envA eyeblink: {np.array(envA_eyeblink).shape}, envB eyeblink: {np.array(envB_eyeblink).shape}, dimensions: {dimensions}")

    output_dimension = dimensions  #here, we set as a variable for hypothesis testing below.
    print(f"[cond_decoding_AvsB] Initializing CEBRA model (output_dim=3, max_iter=1000, lr=4.5e-07)")
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
    shuff_control_all = []
    fract_test_all = []
    shuff_test_all = []

    print(f"[cond_decoding_AvsB] Beginning {NUM_COND_DECODING_REPS} conditioning decoding repetitions")
    for i in range(NUM_COND_DECODING_REPS):

          print(f"[cond_decoding_AvsB] --- Repetition {i+1}/{NUM_COND_DECODING_REPS} ---")

          #test control environment

          ######### use this to test in own environment
          print(f"[cond_decoding_AvsB] [{i+1}] Splitting envA data (75/25 train/test hold-out)")
          eyeblink_train_control, eyeblink_test_control = hold_out(envA_eyeblink, .75)
          cell_train_control, cell_test_control  = hold_out(envA_cell_train,.75)
          print(f"[cond_decoding_AvsB] [{i+1}]   cell_train: {np.array(cell_train_control).shape}, cell_test: {np.array(cell_test_control).shape}")

          #run the model
          print(f"[cond_decoding_AvsB] [{i+1}] Fitting CEBRA model on envA train split")
          cebra_loc_modelpos = sklearn.base.clone(cebra_loc_model).fit(cell_train_control, eyeblink_train_control)
          #determine model fit
          print(f"[cond_decoding_AvsB] [{i+1}] Transforming envA train and test splits")
          cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
          cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)

          #find fraction correct
          fract_controlA = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
          print(f"[cond_decoding_AvsB] [{i+1}] Control (A->A) CSUS score: {fract_controlA:.4f}")



          #test with using A to decode B
          print(f"[cond_decoding_AvsB] [{i+1}] Applying envA model to envB (cross-environment decoding)")
          cell_test = envB_cell_train
          eyeblink_test_control = envB_eyeblink

          #if i want to fit B using fulling training, but i think i want to do it with held out
          '''
          cebra_loc_modelpos_full = cebra_loc_model.fit(envA_cell_train, envA_eyeblink)
          #determine model fit
          cebra_loc_train22 = cebra_loc_modelpos_full.transform(envA_cell_train)
          cebra_loc_test22 = cebra_loc_modelpos_full.transform(cell_test)
          #find fraction correct
          fract_testB = CSUS_score(cebra_loc_train22, cebra_loc_test22, envA_eyeblink, eyeblink_test_control)
          '''

          #determine model fit
          print(f"[cond_decoding_AvsB] [{i+1}] Transforming envB cells with envA-trained model")
          cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test)
          #find fraction correct
          fract_testB = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
          print(f"[cond_decoding_AvsB] [{i+1}] Cross-env (A->B) CSUS score: {fract_testB:.4f}")


          #shuffle
          print(f"[cond_decoding_AvsB] [{i+1}] Building shuffled control: shuffling envA eyeblink labels")
          # Convert to numpy array if not already
          EB = np.array(envA_eyeblink)
          # Create a new array to hold the shuffled data
          EB_shuff = EB.copy()
          # Shuffle each column independently
          #for column in range(EB_shuff.shape[0]):
          np.random.shuffle(EB_shuff[:])

          print(f"[cond_decoding_AvsB] [{i+1}] Splitting shuffled envA data (75/25)")
          eyeblink_train_control, eyeblink_test_control = hold_out(EB_shuff, .75)
          cell_train_control, cell_test_control  = hold_out(envA_cell_train,.75)

          #run the model
          print(f"[cond_decoding_AvsB] [{i+1}] Fitting CEBRA model on shuffled envA labels")
          cebra_loc_modelpos = sklearn.base.clone(cebra_loc_model).fit(cell_train_control, eyeblink_train_control)
          #determine model fit
          print(f"[cond_decoding_AvsB] [{i+1}] Transforming shuffled train and test splits")
          cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
          cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)

          #find fraction correct
          fract_controlA = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
          print(f"[cond_decoding_AvsB] [{i+1}] Shuffled control CSUS score: {fract_controlA:.4f}")

          fract_control_all.append(fract_controlA)
          fract_test_all.append(fract_testB)
          shuff_control_all = []
          shuff_test_all = []
          print(f"[cond_decoding_AvsB] [{i+1}] Appended scores. Running totals — control: {fract_control_all}, test: {fract_test_all}")


    print(f"[cond_decoding_AvsB] All {NUM_COND_DECODING_REPS} repetitions complete. Cleaning up memory.")
    del cebra_loc_modelpos, cebra_loc_train22, cebra_loc_test22
    gc.collect()

    print(f"[cond_decoding_AvsB] Final fract_control_all: {fract_control_all}")
    print(f"[cond_decoding_AvsB] Final fract_test_all: {fract_test_all}")
    print((fract_control_all))
    print((fract_test_all))

    return fract_control_all, fract_test_all
