##################
### Parameters ###
##################
import argparse

parser = argparse.ArgumentParser(description='Using models trained with singleDeep to predict new samples phenotypes')

# Define the parameters and default values
parser.add_argument('--inPath', type=str, help='Folder with the input data (output of PrepareData.R script)')
parser.add_argument('--modelFile', type=str, help='Saved model (output of singleDeep.py script)')
parser.add_argument('--sampleColumn', type=str, default='Sample', help='Column in the metadata of clusters with the sample name')
parser.add_argument('--outFile', type=str, default='prediction_results.tsv', help='Name of the output table')
parser.add_argument('--scale', action='store_true', help='Scale the data')

args = parser.parse_args()

# Assign the paramters to variables
inPath = args.inPath
modelFile = args.modelFile
sampleColumn = args.sampleColumn
outFile = args.outFile
scale = args.scale


####################
### Dependencies ###
####################

# Import functions from functions.py
from functions import NeuralNetwork, \
_get_mean_var, \
scale_transform, singleDeep_predict

# Import libraries
import glob
import pandas as pd
import torch
import itertools


#########################
### Environment setup ###
#########################

# Dictionary to store the predictions
testPredictions = {}


# Get the cluster names from the metadata
filesMeta = glob.glob(inPath + '/Metadata*') # Get the metadata file names
clusters = [] # To store the cluster names

# Get the cluster names from the metadata files names, discarding the path
for file in filesMeta:
    cluster = "/".join(file.split('/')[-1:])
    cluster = "_".join(cluster.split('_')[1:])
    cluster = ".".join(cluster.split('.')[:-1])
    clusters.append(cluster)

# Sort the clusters alphabetically
clusters.sort()

# Prepare metadata for samples
metadataSamples = pd.read_table(inPath + '/Phenodata.tsv', index_col=0)


# Load parameters from the trained models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_dict = torch.load(modelFile, map_location=device)


###########################
### Perform predictions ###
###########################

# Predictions are done for each cluster/cell type, each one with the corresponding pretrained model
for cluster in clusters:
    training_cluster = training_dict[cluster]
    
    # Predict only if cluster MCC > 0
    if training_cluster['MCC'] > 0:
        
        print("Analyzing cluster " + cluster)
        
        # Read files
        expression = pd.read_table(inPath + '/' + cluster + '.tsv', index_col=0)
        metadata = pd.read_table(inPath + '/Metadata_' + cluster + '.tsv', index_col=0)
        
        # Predict the labels for each cell
        testPredictions[cluster] = singleDeep_predict(inPath, sampleColumn, expression, metadata, metadataSamples,
                                            cluster, scale, training_cluster)

# Predict samples labels based on the pondered cells predictions
labelsPredicted = {}
for sample in metadataSamples["Sample"]:
    predictionsSample = []
    for cluster in clusters:
        # Only vote if a prediction has been computed for the sample in this cluster
        if sample in testPredictions[cluster].keys():
            nVotes = max(0, round(training_dict[cluster]['MCC'] * 100)) # Range between 0 to 100 votes depending on the cluster performance
            votes = [testPredictions[cluster][sample]] * nVotes # Add the label as many times as votes
            predictionsSample.append(votes)
    predictionsSample = list(itertools.chain.from_iterable(predictionsSample))
    if len(predictionsSample) > 0:
        prediction = max(set(predictionsSample), key=predictionsSample.count)
        labelsPredicted[sample] = prediction
    else:
        sys.exit("ERROR: Prediction not performed due to none of the models has an MCC > 0")
    

######################
### Export results ###
######################

# Create and write a dataframe with the predictions
results_pd = pd.DataFrame.from_dict(labelsPredicted, orient = "index")
results_pd.columns = ['label_predicted']
results_pd.to_csv(outFile, sep="\t")

