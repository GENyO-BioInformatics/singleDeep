##################
### Parameters ###
##################
import argparse

parser = argparse.ArgumentParser(description='singleDeep: prediction of samples phenotypes from scRNA-Seq data')

parser.add_argument('--inPath', type=str, help='Folder with the input data (output of PrepareData.R script)')
parser.add_argument('--resultsPath', type=str, help='Folder to save the generated reports')
parser.add_argument('--resultsFilenames', type=str, default='singleDeep_results', help='Name of the output files')
parser.add_argument('--logPath', type=str, default='./log', help='Folder to save the generated reports')
parser.add_argument('--varColumn', type=str, default='Condition', help='Column in Phenotype.tsv that contains the analyzed variable')
parser.add_argument('--targetClass', type=int, default=1, help='Gene contributions are calculated related to this class (default = 1, i.e. the second alphabetically ordered category)')
parser.add_argument('--contributions', type=str, default='local', help='Calculate local or global contributions')
parser.add_argument('--sampleColumn', type=str, default='Sample', help='Column in the metadata of clusters with the sample name')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--KOuter', type=int, default=5, help='Folds for the outer cross-validation')
parser.add_argument('--KInner', type=int, default=4, help='Folds for the inner cross-validation')
parser.add_argument('--batchProp', type=float, default=0.1, help='Proportion of training samples for mini-batch')
parser.add_argument('--num_epochs', type=int, default=250, help='Maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=30, help='Minimum number of epochs before early stopping')
parser.add_argument('--eps', type=float, default=0.00001, help='Minimum difference between windows for early stopping')
parser.add_argument('--scale', action='store_true', help='Scale the data')
parser.add_argument('--saveModel', action='store_true', help='Save the model to be used for external data prediction')

args = parser.parse_args()

inPath = args.inPath
resultsPath = args.resultsPath
resultsFilenames = args.resultsFilenames
logPath = args.logPath
varColumn = args.varColumn
targetClass = args.targetClass
contributions = args.contributions
sampleColumn = args.sampleColumn
lr = args.lr
KOuter = args.KOuter
KInner = args.KInner
batchProp = args.batchProp
num_epochs = args.num_epochs
min_epochs = args.min_epochs
eps = args.eps
scale = args.scale
saveModel = args.saveModel


####################
### Dependencies ###
####################

# Import functions from functions.py
from functions import train_loop, test_loop, net_train, net_train_whole, NeuralNetwork, \
init_net, train_evaluate, train_whole, _get_mean_var, \
scale_fit_transform, scale_transform, singleDeep_core

# Import libraries
import sys
import torch
import pandas as pd
import os
from random import seed
from random import sample as rdSample
import sklearn.metrics as metr
from sklearn.metrics import matthews_corrcoef
import glob
import itertools
import statistics


#########################
### Environment setup ###
#########################
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.manual_seed(0)

# Create the output folders if necessary
if not os.path.exists(resultsPath):
    os.mkdir(resultsPath)

if not os.path.exists(logPath):
    os.mkdir(logPath)

foldsPerformancePath = resultsPath + "/folds_performance/"

if not os.path.exists(foldsPerformancePath):
    os.mkdir(foldsPerformancePath)

geneContributionsPath = resultsPath + "/gene_contributions/"

if not os.path.exists(geneContributionsPath):
    os.mkdir(geneContributionsPath)

# Prepare variables to store the results
test_Results = {}
cluster_Results = {}
testPredictions = {}
validationMCCs = {}
testContributions = {}
savedModels = {}
resultsPath += "/"

# Get the cluster names
filesMeta = glob.glob(inPath + '/Metadata*')
clusters = []

for file in filesMeta:
	cluster = file.split('_')[-1]
	cluster = cluster.split('.')[0]
	clusters.append(cluster)

clusters.sort()

# Prepare metadata
metadataSamples = pd.read_table(inPath + '/Phenodata.tsv', index_col=0)

## Assign categorical labels to numbers
labels = sorted(list(set(metadataSamples[varColumn])))
labelsDict = {}
x = 0

for label in labels:
	labelsDict[label] = x
	x += 1

metadataSamples["LabelInt"] = metadataSamples[varColumn].map(labelsDict)

genes = pd.read_table(inPath + "/genes.txt", index_col=0).iloc[:,0].tolist()


#############################
### Train neural networks ###
#############################

for cluster in clusters:
    print("Analyzing cluster " + cluster)
    
    # Read files
    expression = pd.read_table(inPath + '/' + cluster + '.tsv', index_col=0)
    metadata = pd.read_table(inPath + '/Metadata_' + cluster + '.tsv', index_col=0)
    metadata["LabelInt"] = metadata[varColumn].map(labelsDict)
    
    # Get the results for all folds of the cluster
    resultsCluster = singleDeep_core(inPath, varColumn, targetClass, contributions,
                                labelsDict, sampleColumn,
                                 expression, metadata, metadataSamples,
                                lr, num_epochs, min_epochs, eps,
                                logPath, KOuter, KInner, batchProp, 
                                labels, genes, cluster, scale, saveModel)
    
    # Store the results
    testPredictions[cluster] = resultsCluster[0]
    validationPredictions = resultsCluster[1]
    testContributions[cluster] = resultsCluster[2]
    cluster_Results[cluster] = resultsCluster[3]
    
    # Store MCC of phenotype prediction for inner CV
    validationSamplesPredictions = {}
    for sample in metadataSamples["Sample"]:
        predictionsSample = []
        for fold in range(1, KOuter+1):
            if sample in validationPredictions[fold].keys():
                predictionsSample.append(validationPredictions[fold][sample])
        if len(predictionsSample) > 0:
            predictionsSample = list(itertools.chain.from_iterable(predictionsSample))
            prediction = max(set(predictionsSample), key=predictionsSample.count)
            validationSamplesPredictions[sample] = prediction

    labelsReal = metadataSamples["LabelInt"].loc[list(validationSamplesPredictions.keys())]
    x = list(labelsReal)
    y = list(validationSamplesPredictions.values())
    validationMCCs[cluster] = matthews_corrcoef(x, y)
    
    # Save the general model
    if saveModel:
        savedModels[cluster] = resultsCluster[4]
        savedModels[cluster]["MCC"] = validationMCCs[cluster]


##############################################
### Predict labels of test data (outer CV) ###
##############################################

# Predict samples labels based on the pondered cells predictions
labelsPredicted = {}
for sample in metadataSamples["Sample"]:
    predictionsSample = []
    for cluster in clusters:
        # Only vote if a prediction has been computed for the sample in this cluster
        if sample in testPredictions[cluster].keys():
            nVotes = max(0, round(validationMCCs[cluster] * 100))
            votes = [testPredictions[cluster][sample]] * nVotes
            predictionsSample.append(votes)
    predictionsSample = list(itertools.chain.from_iterable(predictionsSample))
    if len(predictionsSample) > 0:
        prediction = max(set(predictionsSample), key=predictionsSample.count)
        labelsPredicted[sample] = prediction
    else:
        sys.exit("ERROR: Prediction not performed due to none of the models has an MCC > 0")
    

# Evaluate the predictions
labelsReal = metadataSamples["LabelInt"].loc[list(labelsPredicted.keys())]
x = list(labelsReal)
y = list(labelsPredicted.values())
test_Results = {'accuracy': metr.accuracy_score(x, y),
               'precision': metr.precision_score(x, y, average='macro'),
               'recall': metr.recall_score(x, y, average='macro'),
               'f1': metr.f1_score(x, y, average='macro'),
               'MCC': matthews_corrcoef(x, y)}


######################
### Export results ###
######################

# Write performance table
results_pd = pd.DataFrame.from_dict(test_Results, orient = "index")
results_pd.to_csv(resultsPath + resultsFilenames + "_testResults.tsv", sep="\t")

# Save gene contributions for each cluster
for cluster in clusters:
    contributions_pd = pd.DataFrame(testContributions[cluster])
    contributions_pd.index = genes
    contributions_pd.to_csv(geneContributionsPath + "geneContributions_cluster_" + cluster + ".tsv", sep="\t")

# Save label prediction performance for each cluster and outer CV fold
for cluster in clusters:
    cluster_Results_pd = pd.DataFrame.from_dict(cluster_Results[cluster], orient="index")
    cluster_Results_pd.to_csv(foldsPerformancePath + "cluster_" + cluster + ".tsv", sep="\t")

# Write a table with the mean performance for each cluster across all folds
cluster_Results_Means = {}
for cluster in clusters:
    cluster_Results_Means[cluster] = {}
    for metric in cluster_Results[cluster][fold].keys():
        sumMetric = 0.0
        for fold in range(1, KOuter+1):
                sumMetric += cluster_Results[cluster][fold][metric]
        meanMetric = sumMetric / KOuter
        cluster_Results_Means[cluster][metric] = meanMetric

cluster_Results_Means_pd = pd.DataFrame.from_dict(cluster_Results_Means, orient = "index")
cluster_Results_Means_pd.to_csv(resultsPath + resultsFilenames + "_clusterResults.tsv", sep="\t")

# Save models
if saveModel:
    torch.save(savedModels, resultsPath + resultsFilenames + "_models.pt")

