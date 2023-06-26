# Parameters
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='singleDeep: prediction of samples phenotypes from scRNA-Seq data')

# Add parameters to the parser
parser.add_argument('--inPath', type=str, help='Folder with the input data (output of PrepareData.R script)')
parser.add_argument('--resultsPath', type=str, help='Folder to save the generated reports')
parser.add_argument('--resultsFilenames', type=str, default='singleDeep_results', help='Name of the output files')
parser.add_argument('--logPath', type=str, default='./log', help='Folder to save the generated reports')
parser.add_argument('--varColumn', type=str, default='Condition', help='Column in Phenotype.tsv that contains the analyzed variable')
parser.add_argument('--sampleColumn', type=str, default='Sample', help='Column in the metadata of clusters with the sample name')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--KOuter', type=int, default=5, help='Folds for the outer cross-validation')
parser.add_argument('--KInner', type=int, default=4, help='Folds for the inner cross-validation')
parser.add_argument('--batchProp', type=float, default=0.1, help='Proportion of training samples for mini-batch')
parser.add_argument('--num_epochs', type=int, default=250, help='Maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=30, help='Minimum number of epochs before early stopping')
parser.add_argument('--eps', type=float, default=0.00001, help='Minimum difference between windows for early stopping')
parser.add_argument('--scale', type=bool, default=True, help='Scale the data. Set to False if data is already scaled')
parser.add_argument('--saveModel', type=bool, default=False, help='Save the model to be used for external data prediction')

# Parse the arguments
args = parser.parse_args()

# Access the parameter values
inPath = args.inPath
resultsPath = args.resultsPath
resultsFilenames = args.resultsFilenames
logPath = args.logPath
varColumn = args.varColumn
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

# # fileName = "cdeprob005"
# # inPath = "Synthetic_data/cdeprob005/"
# inPath = "SLE_data/pediatric_dataset/"
# varColumn = "Condition" # Column in Phenotype.tsv that contains the analyzed variable
# # varColumn = "SLEDAI_Group" # Column in Phenotype.tsv that contains the analyzed variable
# # sampleColumn = "Sample" # Column in the metadata of clusters with the sample name
# sampleColumn = "orig.ident" # Column in the metadata of clusters with the sample name
# # logPath = "log/" # Path to save the reports
# logPath = "log_SLE/pruebaScaling/" # Path to save the report
# # resultsPath = "results_SD/" # Folder to save the results
# resultsPath = "results_SLE/"
# # resultsFilenames = "cdeprob005"
# resultsFilenames = "condition"
# # resultsFilenames = "SLEDAI"
# lr = 0.01 # learning rate
# KOuter = 5
# KInner = 4
# batchProp = 0.1
# num_epochs = 250
# min_epochs = 30
# eps = 0.00001
# saveModel=True


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
metadataSamples = pd.read_table(inPath + '/Phenodata.tsv')

# Assign categorical labels to numbers
labels = sorted(list(set(metadataSamples[varColumn])))
labelsDict = {}
x = 0

for label in labels:
	labelsDict[label] = x
	x += 1

metadataSamples["LabelInt"] = metadataSamples[varColumn].map(labelsDict)
genes = pd.read_table(inPath + "/genes.txt")['x'].tolist()

for cluster in clusters:
    print("Analyzing cluster " + cluster)
    
    # Read files
    expression = pd.read_table(inPath + '/' + cluster + '.tsv')
    metadata = pd.read_table(inPath + '/Metadata_' + cluster + '.tsv')
    metadata["LabelInt"] = metadata[varColumn].map(labelsDict)
    
    # Get the results for all folds of the cluster
    resultsCluster = singleDeep_core(inPath, varColumn, labelsDict, 
                                sampleColumn, expression, metadata, metadataSamples,
                                lr, num_epochs, min_epochs, eps,
                                logPath, KOuter, KInner, batchProp, 
                                labels, genes, cluster, scale, saveModel)

    testPredictions[cluster] = resultsCluster[0]
    validationPredictions = resultsCluster[1]
    testContributions[cluster] = resultsCluster[2]
    cluster_Results[cluster] = resultsCluster[3]
    
    # Store MCC of phenotype prediction for CV
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
    
    if saveModel:
        savedModels[cluster] = resultsCluster[4]
        savedModels[cluster]["MCC"] = validationMCCs[cluster]

    # # Save the mean validation MCCs for the inner plots
    # meanMCC = 0.0
    # for foldOut in range(1, KOuter+1):
    #     for foldIn in range(1, KInner+1):
    #         meanMCC += foldsMCCs[foldOut][foldIn]
    # meanMCC /= (KOuter*KInner)
    # validationMCCs[cluster] = meanMCC


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
    

labelsReal = metadataSamples["LabelInt"].loc[list(labelsPredicted.keys())]
x = list(labelsReal)
y = list(labelsPredicted.values())
test_Results = {'accuracy': metr.accuracy_score(x, y),
               'precision': metr.precision_score(x, y, average='macro'),
               'recall': metr.recall_score(x, y, average='macro'),
               'f1': metr.f1_score(x, y, average='macro'),
               'MCC': matthews_corrcoef(x, y)}

# wrongsamples = ['GSM4029909', 'GSM4029932', 'GSM4029935', 'GSM4029901']
# 
# for sample in wrongsamples:
#     print(sample)
#     for cluster in clusters:
#         print(cluster + ": " + str(testPredictions[cluster][sample]))


######################
### Export results ###
######################
results_pd = pd.DataFrame.from_dict(test_Results, orient = "index")
results_pd.to_csv(resultsPath + resultsFilenames + "_testResults.tsv", sep="\t")



# Save gene contributions for each cluster
for cluster in clusters:
    contributions_pd = pd.DataFrame(testContributions[cluster])
    contributions_pd.index = genes
    contributions_pd.to_csv(geneContributionsPath + "geneContributions_cluster_" + cluster + ".tsv", sep="\t")

# Save phenotype prediction performance for each cluster
for cluster in clusters:
    cluster_Results_pd = pd.DataFrame.from_dict(cluster_Results[cluster], orient="index")
    cluster_Results_pd.to_csv(foldsPerformancePath + "cluster_" + cluster + ".tsv", sep="\t")

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


# contributionsMeans = {}
# for cluster in clusters:
#     contributionsCluster = []
#     for fold in range(1, folds+1):
#         contributionsCluster.append(foldContributions[cluster][fold])
#     contributionsMeans[cluster] = [statistics.mean(group) for group in zip(*contributionsCluster)]
#     
# contributions_pd = pd.DataFrame(contributionsMeans)
# contributions_pd.index = genes
# contributions_pd['Mean'] = contributions_pd.mean(axis=1)
# contributions_pd = contributions_pd.sort_values('Mean', ascending=False)
# contributions_pd.to_csv(resultsPath + resultsFilenames + "_geneContributions.tsv", sep="\t")
