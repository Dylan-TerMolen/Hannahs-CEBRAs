import sys
import argparse
import cebra
from cebra import CEBRA
import cebra.helper as cebra_helper
import numpy as np
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
from cond_decoding_AvsB import cond_decoding_AvsB


#script for running CEBRA within the grid search
#how to run:
    #conda activate cebra
    #python cond_decoding_AvsB_script.py traceA_file traceB_file trainingA_file trainingB_file how_many_divisions pretrial_y_or_n

#ex
    #python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_decoding_AvsB_script.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0
#pretrial_y_or_n: 0 for only cs us, 1 for cs us pretrial
#how many divisions you wanted-- for ex,
#pretrial_y_or_n = 1
    # how_many_divisions = 2 will just split between cs and us
                        #= 10 will split CS and US each into 5

# Setup argument parser
parser = argparse.ArgumentParser(description="Run conditional decoding with CEBRA.")
parser.add_argument("traceA", type=str, help="File path for traceA data.")
parser.add_argument("traceB", type=str, help="File path for traceB data.")
parser.add_argument("trainingA", type=str, help="File path for trainingA data.")
parser.add_argument("trainingB", type=str, help="File path for trainingB data.")
parser.add_argument("how_many_divisions", type=int, help="Number of divisions for categorizing data.")
parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")

# Parse arguments
args = parser.parse_args()

# Load and process data using the arguments
# Adjust the 'file' and 'key' or 'columns' parameters as per your data files and structure
traceA = cebra.load_data(file=args.traceA)  # Adjust 'your_key_here' as necessary
traceB= cebra.load_data(file=args.traceB)  # Adjust 'your_key_here' as necessary
trainingA = cebra.load_data(file=args.trainingA)  # Adjust 'your_key_here' as necessary
trainingB = cebra.load_data(file=args.trainingB)  # Adjust 'your_key_here' as necessary


trainingA = trainingA[0, :]
trainingB = trainingB[0, :]
traceA = np.transpose(traceA)
traceB = np.transpose(traceB)
trainingA = trainingA.flatten()
trainingB = trainingB.flatten()


# Logic to divide data based on 'divisions' and 'pretrial'
if args.pretrial_y_or_n == 0:
    traceA = traceA[trainingA > 0]
    trainingA = trainingA[trainingA > 0]

    traceB = traceB[trainingB > 0]
    trainingB = trainingB[trainingB > 0]

else:
    traceA = traceA[trainingA != 0]
    trainingA = trainingA[trainingA != 0]

    traceB = traceB[trainingB != 0]
    trainingB = trainingB[trainingB != 0]


how_many_divisions = args.how_many_divisions
#    if how_many_divisions == 2:
#        trainingA[(trainingA > 0) & (trainingA <= 5)]  = 1
#        trainingA[trainingA > 5] = 2
#        trainingA[trainingA == -1] = 0

#        trainingB[(trainingB > 0) & (trainingB <= 5)]  = 1
#        trainingB[trainingB > 5] = 2
#        trainingB[trainingB == -1] = 0

if how_many_divisions == 2:
    trainingA[(trainingA > 0) & (trainingA <= 6)]  = 1
    trainingA[trainingA > 6] = 2
    trainingA[trainingA == -1] = 0

    trainingB[(trainingB > 0) & (trainingB <= 6)]  = 1
    trainingB[trainingB > 6] = 2
    trainingB[trainingB == -1] = 0

elif how_many_divisions == 5:
    trainingA[(trainingA > 0) & (trainingA <= 2)]  = 1
    trainingA[(trainingA > 2) & (trainingA <= 4)] = 2
    trainingA[(trainingA > 4) & (trainingA <= 6)] = 3
    trainingA[(trainingA > 6) & (trainingA <= 8)] = 4
    trainingA[trainingA > 8] = 5
    trainingA[trainingA == -1] = 0

    trainingA[(trainingB > 0) & (trainingB <= 2)]  = 1
    trainingB[(trainingB > 2) & (trainingB <= 4)] = 2
    trainingB[(trainingB > 4) & (trainingB <= 6)] = 3
    trainingB[(trainingB > 6) & (trainingB <= 8)] = 4
    trainingB[trainingB > 8] = 5
    trainingB[trainingB == -1] = 0

dimensions = how_many_divisions + args.pretrial_y_or_n

# Running the conditional decoding function
fract_control_all, fract_test_all, _, _, _ = cond_decoding_AvsB(traceA, traceB, trainingA, trainingB, dimensions)
