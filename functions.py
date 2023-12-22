# Import libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metr
from sklearn.metrics import matthews_corrcoef
import statistics
from captum.attr import DeepLift
import warnings

def train_loop(dataloader, model, loss_fn, optimizer, epoch, report=False, writer=False):
    running_loss, correct = 0.0, 0
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    
    
    # Log the running loss
    running_loss /= num_batches
    correct /= size 
    
    if report:
        writer.add_scalar('training loss',
                        running_loss / 1000,
                        epoch + 1)
        
        writer.add_scalar('training accuracy',
                        100*correct,
                        epoch + 1)
    
    return([running_loss, correct])


def test_loop(dataloader, model, loss_fn, epoch, report = False, writer=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            MCCtest = matthews_corrcoef(y.cpu().tolist(), pred.argmax(1).cpu().tolist())
    
    test_loss /= num_batches
    correct /= size
    if report:
        writer.add_scalar('test loss',
                test_loss / 1000,
                epoch + 1)
        
        writer.add_scalar('test accuracy',
                100*correct,
                epoch + 1)
          
        writer.add_scalar('test MCC',
                MCCtest,
                epoch + 1)
    
    return([test_loss, correct])


def net_train(net, train_dataloader, test_dataloader, dtype, device, report=False, writer=False, weights=None, lr=0.001,
                num_epochs = 1000, min_epochs = 50, eps = 0.01):
    net.to(dtype=dtype, device=device)
    global scheduler
    
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=1.0,    # default is no learning rate decay
    )

    num_epochs = num_epochs
    min_epochs = min_epochs # Minimum number of epochs before early stopping
    window_size = 5
    sliding_size = 2
    window1 = 100
    window2 = 0
    totalSize = window_size * 2 - sliding_size
    lossList = []
    minLoss = 1 # Save the minimum achieved loss in this variable
    minLossEpoch = min_epochs # Save the optimal epoch
    params_dict = {}
    
    # Train Network
    for t in range(num_epochs):
        resTrain = train_loop(train_dataloader, net, loss_fn, optimizer, t, report, writer)
        resEpoch = test_loop(test_dataloader, net, loss_fn, t, report, writer)
        current_loss = resEpoch[0] / 1000
        lossList.append(current_loss)
        if len(lossList) > totalSize:
            del(lossList[0])
            window1 = sum(lossList[:window_size]) / window_size
            window2 = sum(lossList[(totalSize-window_size):]) / window_size
        if t > min_epochs and current_loss < minLoss:
            minLoss = current_loss
            params_dict = net.state_dict()
            minLossEpoch = t
        if t > min_epochs and (window2 - window1) > eps:
            break
    
    # Restore the model to the minimum loss point
    if len(params_dict) > 0:
        net.load_state_dict(params_dict)
    
    return [net, minLossEpoch]

# Same as net_train, but modified to train on the whole dataset
def net_train_whole(net, train_dataloader, dtype, device, report=False, 
                writer=False, weights=None, lr=0.001,
                num_epochs = 1000, min_epochs = 50, eps = 0.01):
    net.to(dtype=dtype, device=device)
    global scheduler
    
    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=1.0,    # default is no learning rate decay
    )

    num_epochs = num_epochs
    min_epochs = min_epochs # Minimum number of epochs before early stopping
    window_size = 5
    sliding_size = 2
    window1 = 100
    window2 = 0
    totalSize = window_size * 2 - sliding_size
    lossList = []
    
    # Train Network
    for t in range(num_epochs):
        resTrain = train_loop(train_dataloader, net, loss_fn, optimizer, t, report, writer)
        current_loss = resTrain[0] / 1000
        lossList.append(current_loss)
        if len(lossList) > totalSize:
            del(lossList[0])
            window1 = sum(lossList[:window_size]) / window_size
            window2 = sum(lossList[(totalSize-window_size):]) / window_size
        if t > min_epochs and (window2 - window1) > eps:
            break
    
    return net

# Neural network class: 5 fully connected feed-forward NN
class NeuralNetwork(nn.Module):
    def __init__(self, Hs1, Hs2, Hs3, Hs4, outNeurons, nGenes):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(nGenes, Hs1),
            nn.ReLU(),
            nn.Linear(Hs1, Hs2),
            nn.ReLU(),
            nn.Linear(Hs2, Hs3),
            nn.ReLU(),
            nn.Linear(Hs3, Hs4),
            nn.ReLU(),
            nn.Linear(Hs4, outNeurons),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# Return untrained model
def init_net(Hs1 = 500, Hs2 = 250, Hs3 = 125, Hs4 = 50, outNeurons = 2, nGenes = 100):
    model = NeuralNetwork(Hs1, Hs2, Hs3, Hs4, outNeurons, nGenes)
    return model 

# Main training function: train the models, write reports and return the performance
def train_evaluate(train_dataset, test_dataloader, Hs1 = 500, Hs2 = 250, Hs3 = 125, 
                    Hs4 = 50, outNeurons = 2, report=False, weights=None, lr=0.001,
                    num_epochs=1000, min_epochs=50, eps=0.01, logPath="./", 
                    batchSize=10, cluster="1", foldOut=1, foldIn=1, nGenes = 100, device='cpu'):
    torch.manual_seed(0)
    global trained_net
    
    dtype = torch.float
    
    if report:
        writer = SummaryWriter(logPath + '/' + cluster + '_fold' + str(foldOut) + "_" + str(foldIn))
    else:
        writer = False
    
    # Construct the data loader to adapt the batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize,
                                    shuffle=True, drop_last=True)
    
    # Get neural net
    untrained_net = init_net(Hs1, Hs2, Hs3, Hs4, outNeurons = outNeurons, nGenes = nGenes)
    
    # Train
    trainRes = net_train(net=untrained_net, train_dataloader=train_dataloader, test_dataloader = test_dataloader,
                            dtype=dtype, device=device, report=report, writer = writer, weights=weights, lr=lr,
                            num_epochs = num_epochs, min_epochs = min_epochs, eps = eps)
                            
    trained_net = trainRes[0]
    minLossEpoch = trainRes[1]
    
    if report:
        writer.close()
    
    # Return the MCC of the model as it was trained in this run
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = trained_net(X)
            prediction = pred.argmax(1).cpu().tolist()
            MCCtest = matthews_corrcoef(y.cpu().tolist(), prediction)
            loss = loss_fn(pred, y)
            loss_test = loss.item()

    return [MCCtest, loss_test, minLossEpoch]

# Same as train_evaluate, but only for training whole dataset
def train_whole(train_dataset, Hs1 = 500, Hs2 = 250, Hs3 = 125, Hs4 = 50, outNeurons = 2, report=False, weights=None, lr=0.001,
                    num_epochs=1000, min_epochs=50, eps=0.01, logPath="./", batchSize=10, cluster="1", nGenes = 100, device='cpu'):
    torch.manual_seed(0)
    global trained_net_whole
    
    dtype = torch.float
    
    if report:
        writer = SummaryWriter(logPath + '/' + cluster + '_whole')
    else:
        writer = False
    
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, drop_last=True)
    
    untrained_net = init_net(Hs1, Hs2, Hs3, Hs4, outNeurons = outNeurons, nGenes = nGenes)
    
    trained_net_whole = net_train_whole(net=untrained_net, train_dataloader=train_dataloader,
                            dtype=dtype, device=device, report=report, writer = writer, weights=weights, lr=lr,
                            num_epochs = num_epochs, min_epochs = min_epochs, eps = eps)
    
    if report:
        writer.close()
    

# Calculate mean and variance for a numpy array (source: scanpy)
def _get_mean_var(X, *, axis=0):
    mean = np.mean(X, axis=axis, dtype=np.float64)
    mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
    var = mean_sq - mean**2
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var

# Scaling function (source: scanpy)
def scale_fit_transform(
    X,
    *,
    zero_center: bool = True,
    max_value: float = None,
    copy: bool = False,
    return_mean_std: bool = False,
):
    if np.issubdtype(X.dtype, np.integer):
        logg.info(
            '... as scaling leads to float results, integer '
            'input is cast to float, returning copy.'
        )
        X = X.astype(float)

    mean, var = _get_mean_var(X)
    std = np.sqrt(var)
    std[std == 0] = 1
    X -= mean
    X /= std

    if max_value is not None:
        X[X > max_value] = max_value

    if return_mean_std:
        return X, mean, std
    else:
        return X
    
# Same as scale_fit, but using precomputed mean and std
def scale_transform(
    X,
    mean,
    std,
    *,
    zero_center: bool = True,
    max_value: float = None,
    copy: bool = False,
    return_mean_std: bool = False,
):
    if np.issubdtype(X.dtype, np.integer):
        logg.info(
            '... as scaling leads to float results, integer '
            'input is cast to float, returning copy.'
        )
        X = X.astype(float)
    
    X -= mean
    X /= std
    
    # do the clipping
    if max_value is not None:
        X[X > max_value] = max_value

    if return_mean_std:
        return X, mean, std
    else:
        return X

# Main function
# Return a list with the following objects:
# 1) testPredictions, a dict with the predictions for each test sample
# 2) validationPredictions, a dict with the predictions for each outer fold and sample
# 3) testContributions, the gene contributions for each outer and inner fold
# 4) cluster_Results, the label prediction performance for each outer fold
# 5) outModel, the trained model with all the data for external validation

def singleDeep_core(inPath, varColumn, targetClass, contributions,
                    labelsDict, sampleColumn, 
                    expression, metadata, metadataSamples, lr,
                    num_epochs, min_epochs, eps, logPath, 
                    KOuter, KInner, batchProp,
                    labels, genes, cluster, scale, saveModel):
    
    # Fix seed
    torch.manual_seed(0)
    
    # Prepare output dictionaries
    testPredictions = {}
    validationPredictions = {}
    # validationContributions = {}
    testContributions = {}
    cluster_Results = {}
    minLossEpochs = [] # List to store the optimal minimum epochs for external folds
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nGenes = expression.shape[0]
    nCells = expression.shape[1]
    nSamples = metadataSamples.shape[0]
    
    # Outer split of the nested cross-validation
    outerkf = StratifiedKFold(n_splits = KOuter, random_state=123, shuffle=True)
    for foldOut, (train_index, test_index) in enumerate(outerkf.split(np.zeros(nSamples), metadataSamples['LabelInt'].tolist())):
        foldOut = foldOut + 1
        print("Running outer fold " + str(foldOut) + " of " + str(KOuter))
        
        validationPredictions[foldOut] = {}

        metadataSamplesTrainOut = metadataSamples.iloc[train_index,:]
        trainOutSamples = list(metadataSamplesTrainOut["Sample"])
        metadataSamplesTest = metadataSamples.iloc[test_index,:]
        testSamples = list(metadataSamplesTest["Sample"])
        
        metadataTrainOut = metadata[metadata[sampleColumn].isin(trainOutSamples)]
        metadataTest = metadata[metadata[sampleColumn].isin(testSamples)]
        
        cellsTrainOut = list(metadataTrainOut.index.values)
        cellsTest = list(metadataTest.index.values)
        
        expr_trainOut = expression[cellsTrainOut]
        expr_test = expression[cellsTest]
        expr_trainOut = np.array(expr_trainOut).transpose()
        expr_test = np.array(expr_test).transpose()
        
        # Scaling
        if scale:
            expr_trainOut, meanTrain, stdTrain = scale_fit_transform(expr_trainOut, 
                                                                    return_mean_std=True,
                                                                    max_value=10)
            expr_test = scale_transform(expr_test, mean=meanTrain, std=stdTrain,
                                        max_value=10)
        
        # Create Dataset class from preloaded data
        class exprDataset(Dataset):
            def __init__(self, exprFile, annotations_file):
                self.data_labels = annotations_file
                self.exprFile = exprFile
        
            def __len__(self):
                return len(self.data_labels)
        
            def __getitem__(self, idx):
                data = self.exprFile[idx, :]
                data = torch.from_numpy(data).float().to(device)
                label = torch.as_tensor(self.data_labels[idx]).to(device)
                return data, label
        
        # K-fold Cross-validation for evaluating the performance
        nSamplesTrainOut = metadataSamplesTrainOut.shape[0]
        innerkf = StratifiedKFold(n_splits = KInner, random_state=foldOut, shuffle=True)
        for foldIn, (trainIn_index, validation_index) in enumerate(innerkf.split(np.zeros(nSamplesTrainOut), metadataSamplesTrainOut['LabelInt'].tolist())):
            foldIn = foldIn + 1
            print("Running inner fold " + str(foldIn) + " of " + str(KInner))
            
            metadataSamplesTrainIn = metadataSamplesTrainOut.iloc[trainIn_index,:]
            trainInSamples = list(metadataSamplesTrainIn["Sample"])
            metadataSamplesValidation = metadataSamplesTrainOut.iloc[validation_index,:]
            validationSamples = list(metadataSamplesValidation["Sample"])
            
            metadataTrainIn = metadata[metadata[sampleColumn].isin(trainInSamples)]
            metadataValidation = metadata[metadata[sampleColumn].isin(validationSamples)]
            
            cellsTrainIn = list(metadataTrainIn.index.values)
            cellsValidation = list(metadataValidation.index.values)
            
            indexCellsTrainIn = []
            for cell in cellsTrainIn:
                ind = cellsTrainOut.index(cell)
                indexCellsTrainIn.append(ind)

            expr_trainIn = expr_trainOut[indexCellsTrainIn]
            
            indexCellsValidation = []
            for cell in cellsValidation:
                ind = cellsTrainOut.index(cell)
                indexCellsValidation.append(ind)
                
            expr_validation = expr_trainOut[indexCellsValidation]
            
            
            exprLabels_trainIn = np.array(metadata.loc[cellsTrainIn]['LabelInt'])
            exprLabels_validation = np.array(metadata.loc[cellsValidation]['LabelInt'])
            
            trainIn_dataset = exprDataset(expr_trainIn, exprLabels_trainIn)
            validation_dataset = exprDataset(expr_validation, exprLabels_validation)
            validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)
    
            # Assign weights for imbalanced classes
            weights = []
            
            for label in labelsDict.values():
            	count = metadataTrainIn['LabelInt'].tolist().count(label)
            	w = 1/(count/metadataTrainIn.shape[0])
            	weights.append(w)
            
            weightsScaled = []
            for weight in weights:
            	w = weight/sum(weights)
            	weightsScaled.append(w)
            
            weights=torch.as_tensor(weightsScaled).to(device)
            
            # Define the neurons in each layer
            layer1 = round(nGenes/2)
            layer2 = round(layer1/2)
            layer3 = round(layer2/2)
            layer4 = round(layer3/4)
            outNeurons = len(labels)
            
            batchSize = round(len(expr_trainIn)*batchProp)
            
            resultsValidation = train_evaluate(trainIn_dataset, validation_dataloader, layer1, layer2, layer3, layer4, outNeurons, 
                        report=True, weights=weights, lr=lr, num_epochs = num_epochs, 
                        min_epochs=min_epochs, eps=eps, logPath=logPath, batchSize=batchSize, 
                        cluster=cluster, foldOut=foldOut, foldIn = foldIn, nGenes=nGenes, device=device)
            
            
            # Predict the labels of the validation samples
            for sample in validationSamples:
                metadataSample = metadataValidation[metadataValidation[sampleColumn].isin([sample])]
                # Predict only if there are cells of the sample in this cluster
                if metadataSample.shape[0] > 0:
                    cellsSample = list(metadataSample.index.values)
                    indexCells = []
                    for cell in cellsSample:
                        ind = cellsValidation.index(cell)
                        indexCells.append(ind)
                    exprSample = expr_validation[indexCells]
                    exprSampleTensor = torch.from_numpy(exprSample).float().to(device)
                    pred = trained_net(exprSampleTensor)
                    labelsPred = pred.argmax(1).cpu().tolist()
                    # prediction = max(set(labelsPred), key=labelsPred.count)
                    validationPredictions[foldOut][sample] = labelsPred
                    
            # End of inner loop
        
        exprLabels_trainOut = np.array(metadata.loc[cellsTrainOut]['LabelInt'])
        exprLabels_test = np.array(metadata.loc[cellsTest]['LabelInt'])
        
        trainOut_dataset = exprDataset(expr_trainOut, exprLabels_trainOut)
        test_dataset = exprDataset(expr_test, exprLabels_test)
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        # Assign weights for imbalanced classes
        weights = []
        
        for label in labelsDict.values():
        	count = metadataTrainOut['LabelInt'].tolist().count(label)
        	w = 1/(count/metadataTrainOut.shape[0])
        	weights.append(w)
        
        weightsScaled = []
        for weight in weights:
        	w = weight/sum(weights)
        	weightsScaled.append(w)
        
        weights=torch.as_tensor(weightsScaled).to(device)
        
        # Define the neurons in each layer
        layer1 = round(nGenes/2)
        layer2 = round(layer1/2)
        layer3 = round(layer2/2)
        layer4 = round(layer3/4)
        outNeurons = len(labels)
        
        batchSize = round(len(expr_trainOut)*batchProp)
        
        resultsTest = train_evaluate(trainOut_dataset, test_dataloader, layer1, layer2, layer3, layer4, outNeurons, 
                                report=True, weights=weights, lr=lr, num_epochs = num_epochs, 
                                min_epochs=min_epochs, eps=eps, logPath=logPath, batchSize=batchSize, 
                                cluster=cluster, foldIn="test", foldOut=foldOut, nGenes=nGenes, device=device)
        
        minLossEpochs.append(resultsTest[2])
        
        # Make predictions for test cells
        for sample in testSamples:
            metadataSample = metadata[metadata[sampleColumn].isin([sample])]
            # Predict only if there are cells of the sample in this cluster
            if metadataSample.shape[0] > 0:
                cellsSample = list(metadataSample.index.values)
                indexCells = []
                for cell in cellsSample:
                    ind = cellsTest.index(cell)
                    indexCells.append(ind)
                # Prediction
                exprSample = expr_test[indexCells]
                exprSampleTensor = torch.from_numpy(exprSample).float().to(device)
                pred = trained_net(exprSampleTensor)
                labelsPred = pred.argmax(1).cpu().tolist()
                prediction = max(set(labelsPred), key=labelsPred.count)
                testPredictions[sample] = prediction
                
            
            # Performance of label prediction for this fold
            labelsReal = metadataSamples["LabelInt"].loc[list(testPredictions.keys())]
            x = list(labelsReal)
            y = list(testPredictions.values())
            performance = {'accuracy': metr.accuracy_score(x, y),
                       'precision': metr.precision_score(x, y, average='macro'),
                       'recall': metr.recall_score(x, y, average='macro'),
                       'f1': metr.f1_score(x, y, average='macro'),
                       'MCC': matthews_corrcoef(x, y)}
            cluster_Results[foldOut] = performance
            
            # End of outer loop
        
        if device == "cuda":
            torch.cuda.empty_cache()
            
    # Train the model with all data
    outModel = {}
    exprLabels = np.array(metadata['LabelInt'])
    expression = np.array(expression).transpose()
    # Scaling
    if scale:
        expression, meanExpr, stdExpr = scale_fit_transform(expression,
                                                            return_mean_std=True,
                                                            max_value=10)
        outModel["mean"] = meanExpr
        outModel["std"] = stdExpr
    
    expression_dataset = exprDataset(expression, exprLabels)
    
    # Assign weights for imbalanced classes
    weights = []
    for label in labelsDict.values():
    	count = exprLabels.tolist().count(label)
    	w = 1/(count/metadata.shape[0])
    	weights.append(w)
    
    weightsScaled = []
    for weight in weights:
    	w = weight/sum(weights)
    	weightsScaled.append(w)
    
    weights=torch.as_tensor(weightsScaled).to(device)
    
    # Define the neurons in each layer
    layer1 = round(nGenes/2)
    layer2 = round(layer1/2)
    layer3 = round(layer2/2)
    layer4 = round(layer3/4)
    outNeurons = len(labels)
    
    batchSize = round(len(expression)*batchProp)
    
    epochsWhole = round(sum(minLossEpochs) / len(minLossEpochs))
    
    train_whole(expression_dataset, layer1, layer2, layer3, layer4, outNeurons, 
                report=True, weights=weights, lr=lr, num_epochs = epochsWhole, 
                min_epochs=epochsWhole, eps=eps, logPath=logPath, batchSize=batchSize, 
                cluster=cluster, nGenes=nGenes, device=device)
    
    # Calculate genes contributions
    # Define the baseline as the minimum expression for each gene
    minExpressionGenes = expression.min(axis=0)
    baseline = torch.from_numpy(np.tile(minExpressionGenes, (len(expression_dataset), 1))).float().to(device)
    
    if contributions == "local":
        dl = DeepLift(trained_net_whole, multiply_by_inputs = False)
    else:
        dl = DeepLift(trained_net_whole, multiply_by_inputs = True)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        contributions = dl.attribute(torch.from_numpy(expression).float().to(device), baseline,
                                            target=targetClass)
    
    samples = list(metadataSamples["Sample"])
    cells = list(metadata.index.values)
    
    for sample in samples:
        metadataSample = metadata[metadata[sampleColumn].isin([sample])]
        # Predict only if there are cells of the sample in this cluster
        if metadataSample.shape[0] > 0:
            cellsSample = list(metadataSample.index.values)
            indexCells = []
            for cell in cellsSample:
                ind = cells.index(cell)
                indexCells.append(ind)
            # Mean contributions
            contributionSample = contributions[indexCells]
            contributionSample_means = contributionSample.mean(dim=0).to("cpu").tolist()
            testContributions[sample] = contributionSample_means
            # contributionSample = contributionSample.to("cpu").detach().numpy()
            # max_abs_index = np.argmax(np.abs(contributionSample), axis=0)
            # contributionSampleMax = contributionSample[max_abs_index, np.arange(contributionSample.shape[1])]
            # testContributions[sample] = contributionSampleMax

                
    outModel["model"] = trained_net_whole.state_dict()
        
    return([testPredictions, validationPredictions, testContributions, cluster_Results, outModel])


# Predict samples using a previously trained model
def singleDeep_predict(inPath, sampleColumn, 
                        expression, metadata, metadataSamples,
                        cluster, scale, training_cluster):
    
    # Prepare output dictionaries
    testPredictions = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nGenes = expression.shape[0]
    nCells = expression.shape[1]
    nSamples = metadataSamples.shape[0]
    trained_parameters = training_cluster['model']
    
    samples = list(metadataSamples["Sample"])
    cells = list(metadata.index.values)
    expression = np.array(expression).transpose()
    
    # Scaling
    if scale:
        expression = scale_transform(expression, mean=training_cluster['mean'], 
                                        std=training_cluster['std'],
                                        max_value=10)
        
    # Define the neurons in each layer
    layer1 = round(nGenes/2)
    layer2 = round(layer1/2)
    layer3 = round(layer2/2)
    layer4 = round(layer3/4)
    outNeurons = len(trained_parameters['linear_relu_stack.8.bias'])
    
    trained_net = NeuralNetwork(layer1, layer2, layer3, layer4, outNeurons = outNeurons, nGenes = nGenes)
    trained_net.load_state_dict(trained_parameters)
    trained_net.to(dtype=torch.float, device=device)

    # Make predictions for cells
    for sample in samples:
        metadataSample = metadata[metadata[sampleColumn].isin([sample])]
        # Predict only if there are cells of the sample in this cluster
        if metadataSample.shape[0] > 0:
            cellsSample = list(metadataSample.index.values)
            indexCells = []
            for cell in cellsSample:
                ind = cells.index(cell)
                indexCells.append(ind)
            # Prediction
            exprSample = expression[indexCells]
            exprSampleTensor = torch.from_numpy(exprSample).float().to(device)
            pred = trained_net(exprSampleTensor)
            labelsPred = pred.argmax(1).cpu().tolist()
            prediction = max(set(labelsPred), key=labelsPred.count)
            testPredictions[sample] = prediction
        
    if device == "cuda":
        torch.cuda.empty_cache()

    return(testPredictions)


