#!/home/genyo/anaconda3/bin/python3.8

# Parameters
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Using models trained with singleDeep to predict new samples phenotypes')

# Add parameters to the parser
parser.add_argument('--inPath', type=str, help='Folder with the input data (output of PrepareData.R script)')
parser.add_argument('--modelFile', type=str, help='Saved model (output of singleDeep.py script)')
parser.add_argument('--sampleColumn', type=str, default='Sample', help='Column in the metadata of clusters with the sample name')
parser.add_argument('--scale', type=bool, default=True, help='Scale the data. Set to False if data is already scaled')
parser.add_argument('--outFile', type=str, default='prediction_results.tsv', help='Name of the output table')

# Parse the arguments
args = parser.parse_args()

# Access the parameter values
inPath = args.inPath
modelFile = args.modelFile
sampleColumn = args.sampleColumn
scale = args.scale
outFile = args.outFile

inPath = "SLE_data/pediatric_scaled_projected/"
modelFile = "results_SLEScience3/Status_models.pt"
sampleColumn="ind_cov"
varColumn = "Status"
outFile = "results_SLEScience3/validation_predictions.tsv"
scale = False

# Import functions from functions.py
from functions import NeuralNetwork, \
_get_mean_var, \
scale_transform, singleDeep_predict

# Import libraries
import glob
import pandas as pd
import torch
import itertools



# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# torch.manual_seed(0)





test_Results = {}
cluster_Results = {}
testPredictions = {}


# Get the cluster names
filesMeta = glob.glob(inPath + '/Metadata*')
clusters = []

for file in filesMeta:
	cluster = file.split('_')[-1]
	cluster = cluster.split('.')[0]
	clusters.append(cluster)

clusters.sort()

metadataSamples = pd.read_table(inPath + '/Phenodata.tsv')

# Remove from here
# Assign categorical labels to numbers
labels = list(set(metadataSamples[varColumn]))
labelsDict = {}
x = 1

for label in labels:
	labelsDict[label] = x
	x -= 1

metadataSamples["LabelInt"] = metadataSamples[varColumn].map(labelsDict)

# Assign categorical labels to numbers
# genes = pd.read_table(inPath + "/genes.txt")['x'].tolist()

# Load training parameters
training_dict = torch.load(modelFile)


for cluster in clusters:
    training_cluster = training_dict[cluster]
    
    # Predict only if cluster MCC > 0
    if training_cluster['MCC'] > 0:
        
        print("Analyzing cluster " + cluster)
        
        # Read files
        expression = pd.read_table(inPath + '/' + cluster + '.tsv')
        metadata = pd.read_table(inPath + '/Metadata_' + cluster + '.tsv')
        
        # Predict the labels for each cell
        testPredictions[cluster] = singleDeep_predict(inPath, sampleColumn, expression, metadata, metadataSamples,
                                            cluster, scale, training_cluster)


# # Print Accuracy and MCC for each cluster
# import sklearn.metrics as metr
# from sklearn.metrics import matthews_corrcoef
# for cluster in clusters:
#     print(cluster)
#     x = list(metadataSamples["LabelInt"].loc[list(testPredictions[cluster].keys())])
#     y = list(testPredictions[cluster].values())
#     metr.accuracy_score(x, y)
#     metr.matthews_corrcoef(x, y)

# Predict samples labels based on the pondered cells predictions
labelsPredicted = {}
for sample in metadataSamples["Sample"]:
    predictionsSample = []
    for cluster in clusters:
        # print(cluster)
        # print(training_dict[cluster]['MCC'])
        # Only vote if a prediction has been computed for the sample in this cluster
        if sample in testPredictions[cluster].keys():
            nVotes = max(0, round(training_dict[cluster]['MCC'] * 100))
            votes = [testPredictions[cluster][sample]] * nVotes
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
results_pd = pd.DataFrame.from_dict(labelsPredicted, orient = "index")
results_pd.columns = ['label_predicted']
results_pd.to_csv(outFile, sep="\t")
