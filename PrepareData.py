import argparse
import anndata
import pandas as pd
import numpy as np
import os
import random

# Define argument parser
parser = argparse.ArgumentParser(description="Process arguments.")
parser.add_argument("--inputPath", type=str, default=None, help="Path of the input object. Seurat object in rds file or scanpy object in h5ad file")
parser.add_argument("--sampleColumn", type=str, default=None, help="Metadata column with samples labels.")
parser.add_argument("--clinicalColumns", type=str, default=None, help="Specify the columns with clinical information. If there are more than one columns, separate the names by ','. Example: col1,col2")
parser.add_argument("--clusterColumn", type=str, default=None, help="Specify the columns with cluster information")
parser.add_argument("--minCellsSample", type=int, default=0, help="Minimum number of cells that must contain each sample in each cluster")
parser.add_argument("--minCells", type=int, default=100, help="Minimum number of cells per cluster. If the number of cells is lower than this, the cluster is removed")
parser.add_argument("--maxCells", type=int, default=50000, help="Maximum number of cells per cluster. If the number of cells is higher than this, random cells are removed")
parser.add_argument("--outPath", type=str, default="outData", help="Output path")
args = parser.parse_args()

# Default values
if args.sampleColumn is None:
    args.sampleColumn = "ind_cov"
if args.clinicalColumns is None:
    args.clinicalColumns = "Status"
if args.clusterColumn is None:
    args.clusterColumn = "louvain"

# Load data
adata = anndata.read_h5ad(args.inputPath)
exprMatrix = adata.X
metadataCells = adata.obs

# Extract unique samples and clusters
samples = metadataCells[args.sampleColumn].unique()
clusters = metadataCells[args.clusterColumn].unique()

# Extract clinical metadata
clinicalColumns = args.clinicalColumns.split(',')
metadataSamples = pd.DataFrame(columns=clinicalColumns + ['Sample'])
for sample in samples:
    sample_metadata = metadataCells.loc[metadataCells[args.sampleColumn] == sample, clinicalColumns].iloc[0].tolist()
    metadataSamples.loc[sample] = sample_metadata + [sample]

# Calculate the number of samples with a minimum number of cells in each cluster
clusterSamples = {}
clusterNCells = {}
for cluster in clusters:
    metaCluster = metadataCells[metadataCells[args.clusterColumn] == cluster]
    nCellsCluster = [sum(metaCluster[args.sampleColumn] == sample) for sample in samples]
    nSamplesPass = sum(np.array(nCellsCluster) >= args.minCellsSample)
    clusterSamples[cluster] = nSamplesPass
    clusterNCells[cluster] = metaCluster.shape[0]

# Save data for singleDeep
clustersOK = [cluster for cluster, n_samples in clusterSamples.items() if n_samples == len(samples) and clusterNCells[cluster] >= args.minCells]
os.makedirs(args.outPath, exist_ok=True)
random.seed(123)
for cluster in clustersOK:
    metaCluster = metadataCells[metadataCells[args.clusterColumn] == cluster]
    cellsCluster = metaCluster.index
    # Cells subsampling
    maxCells = min(args.maxCells, len(cellsCluster))
    if len(cellsCluster) > args.maxCells:
        cellsCluster = random.sample(cellsCluster.tolist(), maxCells)
        metaCluster = metaCluster.loc[cellsCluster]
    exprCluster = adata[adata.obs_names.isin(cellsCluster),:]
    exprCluster = exprCluster.to_df().T
    exprCluster.index.name=None
    exprCluster = exprCluster.rename_axis(None, axis=1)
    exprCluster.to_csv(os.path.join(args.outPath, f"{cluster}.tsv"), sep="\t", index=True, header=True)
    metaCluster.index.name=None
    metaCluster = metaCluster.rename_axis(None, axis=1)
    metaCluster.to_csv(os.path.join(args.outPath, f"Metadata_{cluster}.tsv"), sep="\t", index=True, header=True)

metadataSamples.to_csv(os.path.join(args.outPath, "Phenodata.tsv"), sep="\t", index=True, header=True)

genes = pd.DataFrame(adata.var_names)
genes.to_csv(os.path.join(args.outPath, "genes.txt"), sep="\t", index=True, header=True)

