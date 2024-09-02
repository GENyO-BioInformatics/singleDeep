# singleDeep

A deep learning workflow for samples phenotype prediction with single-cell RNA-Seq data.

## Dependencies

SingleDeep has been tested with R 4.2.0 and Python 3.8.3, although it is likely to work with other versions. To install the Python dependencies with pip, run the following code.

```{bash}
pip install -r requirements.txt
```

## Preparing the data

The starting point of singleDeep is a Seurat or scanpy object with the scRNA-Seq data processed and the cell populations annotated. The file format should be RData or h5ad for Seurat and scanpy objects respectively. There are several resources to process scRNA-Seq data, for instance the excellent book Single-cell best practices (<https://www.sc-best-practices.org/>). Scaling is optional, but it is important to know whether it has been performed or not, since it has to be indicated to the singleDeep main script (parameter *scale*). To reduce the computational cost, it is strongly recommended to perform some gene selection during the data preprocessing. For instance, it is common to select a number of highly variable genes.

You can run the R script **PrepareData.R** from the terminal to read the input file and prepare all the necessary data to run singleDeep. It is strongly recommended to read the script help to understand the parameters and be used adequately. You can access this help with the following command.

```{bash}
Rscript PrepareData.R --help
```

Among the script parameters, there are two specially relevant:

-   ***maxCells***: If a cell type has a large amount of cells (e.g., \> 100,000), there is no benefit for the neural networks training and the computational cost is significantly increased. For this reason, it is recommended to use this parameter to discard random cells until reaching the specified value (50,000 by default, but usually less cells can be used without affecting the models performance).

-   ***filterGenes***: If this parameter is included, ribosomal, mitochondrial and non-coding genes will be discarded from the data. These genes are usually not measured correctly in single-cell experiments and may be problematic for data analysis and interpretation. Therefore, we recommend to include this parameter unless there is some specific interest for these genes.

Here is an example of how to run the script for a scanpy object:

```{bash}
Rscript PrepareData.R --inputPath dataset.h5ad --fileType scanpy --sampleColumn ind_cov --clusterColumn cg_cov --clinicalColumns Disease --maxCells 30000 --outPath ./singleDeep_input

```

## Training and internal validation

This is the main step of the singleDeep workflow. Check carefully the parameters of **singleDeep.py**, since many of them are essential for the analyses to be run correctly. This script trains a deep learning model for each cell population previously annotated. Internal validation is performed with a nested cross-validation, with the inner loop used to calculate the performance of the cluster (MCC parameter) and the outer loop performs the actual training and prediction for the individual cells. Based on these per-cell predictions, a per-sample prediction is performed for each cell population. A final prediction considering all the cell populations is calculated, and the performance metrics are calculated comparing these predictions and the real labels provided.

To read the script help, run the following command:

```{bash}
python singleDeep.py --help
```

Following the example of PrepareData.R, this script would be executed with default parameters using the following command.

```{bash}
python singleDeep.py --inPath ./singleDeep_input
```

The output files, saved by default in the folder *results*, are the following:

-   **testResults.tsv**: Mean performance metrics for the test samples (outer fold of the nested cross-validation)

-   **clusterResults.tsv**: Mean performance metrics for the individual cell populations. This is useful to know which clusters are the most informative for the classification

-   **folds_performance**: Folder with performance metrics for each cell population and individual external fold. These tables give information about the performance consistency along the different data folds in each cell population

-   **gene_contributions**: Folder with the contributions of each gene in each sample and cell population. Read the DeepLIFT article for more information (<https://arxiv.org/abs/1704.02685>)

-   **models.pt** (only if --saveModel parameter is passed): File necessary for external validation

## Training reports

SingleDeep saves interactive reports that can be used to track the training and validation processes. It is necessary to install TensorFlow to open these reports (check <https://www.tensorflow.org/install>). Once installed, the reports may be opened with this command:

```{bash}
tensorboard --logdir ./log
```

The reports to be plotted may be selected in the left panel. The structure of the reports names for the inner cross-validation are *cell population + fold K out + fold K in.* For instance, the first inner fold for the second outer fold and B cell population would be *B_fold2_1*. The outer cross-validation is named as *test* (e.g. *B_fold3_test* would contain the results for the third fold of the outer loop). Furthermore, if the sameModel parameter of the singleDeep.py script is True, reports finishing with *whole* are generated, containing the training metrics for the entire dataset.

Reports contain the MCC, accuracy and cross-entropy loss for both the test and training along the training epochs.

## Using the trained models with external data

If the parameter *saveModel* of singleDeep.py is set to True, the neural networks for each cell population are trained with the whole dataset and saved to the **models.pt** file, together with other necessary data. To use these models to predict the phenotype of a new dataset, it is necessary to project the cells of this dataset to the training dataset. This step is essential to apply the model of each cell population to the corresponding cells. There are several methods to do this step, such as scmap (<https://bioconductor.org/packages/release/bioc/html/scmap.html>) in R or the scanpy's function tl.ingest (<https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.ingest.html>). Furthermore, according to our experience, it is also necessary to correct the batch effect between the datasets with methods such as Combat or Harmony, as well as scaling together the datasets and selecting the variable genes from the merged data.

The function **PrepareData.R** may be used on this new dataset following the previous instructions. Once these data are ready, the script **singleDeep_pretrained.py** should be used to perform the prediction. You can check the help with the following code:

```{bash}
python singleDeep_pretrained.py --help
```

Essentially, the required parameters are *inputPath*, the path to the new dataset (output of PrepareData.R); *modelFile*, the object with the trained models (models.pt from singleDeep.py); *scale* to indicate if scaling should be performed; and *outFile*, the file to save the predictions. Take into account that singleDeep transforms the phenotype categories into numbers (0, 1...) following the alphabetical order. For instance, if the caregories are "Control" and "Disease", the output will be 0 for Control and 1 for Disease respectively.

## Use example

We created a toy dataset comprising 20 samples, each containing 5 distinct cell types under 2 different conditions. You can generate this dataset using the `simulation_toy_dataset.R` script. To run the singleDeep model on this simulated data, use the following command:

```{bash}
python singleDeep.py --inPath ./toy_dataset/data --sampleColumn Sample --logPath log_toy --resultsPath results_toy --varColumn Condition --num_epochs 50 --resultsFilenames Condition --KOuter 3 --KInner 2
```

Please note that you may encounter 'zero_division' warnings, which are expected if all samples are predicted with the same label. The process will take a few minutes to complete, after which the results will be available in the `results_toy` folder.
