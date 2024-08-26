# Import the necessary packages
import argparse
import anndata
import pandas as pd
import numpy as np
import os
import random
from collections import Counter
from pybiomart import Dataset
import re

# Define argument parser
parser = argparse.ArgumentParser(description="Process arguments.")
parser.add_argument("--inputPath", type=str, default=None, help="Path of the input object. Seurat object in rds file or scanpy object in h5ad file")
parser.add_argument("--sampleColumn", type=str, default=None, help="Metadata column with samples labels.")
parser.add_argument("--clinicalColumns", type=str, default=None, help="Specify the columns with clinical information. If there are more than one columns, separate the names by ','. Example: col1,col2")
parser.add_argument("--targetColumn", type=str, default=None, help="Specify the column with the class to be predicted")
parser.add_argument("--clusterColumn", type=str, default=None, help="Specify the columns with cluster information")
parser.add_argument("--filterGenes", action="store_true", default=False, help="Filter non-coding, mitochondrial, ribosomal and hemoglobulin genes (recommended)")
parser.add_argument("--organism", type=str, default="hsapiens", help="BiomaRt organism to annotate the genes for filtering"),
parser.add_argument("--minCellsSample", type=int, default=0, help="Minimum number of cells that must contain each sample in each cluster")
parser.add_argument("--minCellsClass", type=int, default=10, help="Minimum number of cells that must contain each target class in each cluster")
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

# Calculate the number of samples with the specified minimum of cells in each cluster
clusterSamples = {}
clusterClassCells = {}
clusterNCells = {}
for cluster in clusters:
    metaCluster = metadataCells[metadataCells[args.clusterColumn] == cluster]
    nCellsCluster = [sum(metaCluster[args.sampleColumn] == sample) for sample in samples]
    nSamplesPass = sum(np.array(nCellsCluster) >= args.minCellsSample)
    clusterSamples[cluster] = nSamplesPass
    emptyCounter = {}
    for classTarget in metadataCells[args.targetColumn].unique():
        emptyCounter[classTarget] = 0
    emptyCounter = Counter(emptyCounter)
    emptyCounter.update(metaCluster[args.targetColumn].tolist())
    clusterClassCells[cluster] = min(emptyCounter.values())
    clusterNCells[cluster] = metaCluster.shape[0]


# Gene filtering
# Get the gene names from the expression data
genes = list(adata.var_names)

if args.filterGenes:
    # Get gene annotation
    dataset = args.organism + "_gene_ensembl"
    mart = Dataset(name=dataset, host="http://www.ensembl.org")
    annot = mart.query(attributes=["external_gene_name", "gene_biotype"])
    # Find mitochondrial, ribosomal, hemoglobin and non-coding genes
    mitGenes = [gene for gene in genes if re.match("^MT-", gene.upper())]
    ribGenes = [gene for gene in genes if re.match("^RPL", gene.upper())]
    ribGenes.extend([gene for gene in genes if re.match("^RPS", gene.upper())])
    hbGenes = [gene for gene in genes if re.match("^HB", gene.upper())]
    nonCodingGenes = annot[annot["Gene type"] != "protein_coding"]
    nonCodingGenes = nonCodingGenes["Gene name"].tolist()
    listExclusion = mitGenes + ribGenes + hbGenes + nonCodingGenes
    # Discard the previous genes from the expression data
    exprMatrix = exprMatrix[:, ~np.isin(genes, listExclusion)]


# Save data for singleDeep

# Define the clusters that pass the stablished thresholds
clustersOK = [cluster for cluster, n_samples in clusterSamples.items() if n_samples == len(samples) and clusterClassCells[cluster] >= args.minCellsClass and clusterNCells[cluster] >= args.minCells]
os.makedirs(args.outPath, exist_ok=True)
random.seed(123)

# Subsample cells if necessary and save the gene expression and metadata tables
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
    cluster = cluster.replace("/", "_")
    exprCluster.to_csv(os.path.join(args.outPath, f"{cluster}.tsv"), sep="\t", index=True, header=True)
    metaCluster.index.name=None
    metaCluster = metaCluster.rename_axis(None, axis=1)
    metaCluster.columns = metaCluster.columns.str.replace(' ', '_')
    metaCluster.to_csv(os.path.join(args.outPath, f"Metadata_{cluster}.tsv"), sep="\t", index=True, header=True)


metadataSamples.columns = metadataSamples.columns.str.replace(' ', '_')
metadataSamples.to_csv(os.path.join(args.outPath, "Phenodata.tsv"), sep="\t", index=True, header=True)

if args.filterGenes:
    genes = [gene for gene in genes if gene not in listExclusion]
else:
    genes = adata.var_names

# Save the file with the gene names
genes = pd.DataFrame(genes)
genes.to_csv(os.path.join(args.outPath, "genes.txt"), sep="\t", index=True, header=True)

