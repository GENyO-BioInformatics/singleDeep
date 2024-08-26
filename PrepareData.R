# Install optparse package to parse the parameters
if (!require("optparse", character.only = TRUE)) {
    install.packages("optparse", dependencies = TRUE)
    library("optparse", character.only = TRUE)
}

# Parameters definiton
option_list <- list(
    make_option(c("--inputPath"), type="character", default = NULL,
                help="Path of the input object. Seurat object in rds file or scanpy
        object in h5ad file"),
    make_option(c("--fileType"), type="character", default="seurat",
                help="File type (seurat or scanpy)"),
    make_option(c("--slotSeurat"), type="character", default= NULL,
                help="Only when fileType == seurat. Slot of the Seurat object
        to retrieve expression (data, scale.data, counts)"),
    make_option(c("--sampleColumn"), type="character", default= NULL,
                help="Metadata column with samples labels."),
    make_option(c("--clinicalColumns"), type="character",
                default= NULL,
                help="Specify the columns with clinical information.
        If there are more than one columns,
        separate the names by ','. Example: col1,col2"),
    make_option(c("--targetColumn"), type="character",
                default= NULL,
                help="Specify the column with the class to be predicted"),
    make_option(c("--clusterColumn"), type="character", default= NULL,
                help="Specify the columns with cluster information"),
    make_option(c("--filterGenes"), type="logical", action="store_true", default = FALSE,
                help="Filter non-coding, mitochondrial, ribosomal and hemoglobulin genes (recommended)"),
    make_option(c("--organism"), type="character", default= "hsapiens",
                help="BiomaRt organism to annotate the genes for filtering"),
    make_option(c("--minCellsSample"), type="numeric", default= 0,
                help="Minimum number of cells that must contain
        each sample in each cluster"),
    make_option(c("--minCellsClass"), type="numeric", default= 10,
                help="Minimum number of cells that must contain
        each target class in each cluster"),
    make_option(c("--minCells"), type="numeric", default= 100,
                help="Minimum number of cells per cluster. If the number of
        cells is lower than this, the cluster is removed"),
    make_option(c("--maxCells"), type="numeric", default= 50000,
                help="Maximum number of cells per cluster. If the number of
        cells is higher than this, random cells are removed"),
    make_option(c("--outPath"), type="character", default= "outData",
                help="Output path"),
    make_option(c("--pythonPath"), type="character", default= NULL,
                help="Path to the Python executables. Only if fileType == scanpy")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Checks
if(!(opt$fileType== "scanpy" | opt$fileType == "seurat")){
    stop("It has not been specified if the file type is scanpy or seurat")
}

# Default slotSeurat to scale.data if not defined
if(opt$fileType == "seurat" & is.null(opt$slotSeurat)){
    opt$slotSeurat <- "scale.data"
}

#Default columns in Seurat objects if not defined
if(opt$fileType == "seurat" & is.null(opt$sampleColumn)){
    opt$sampleColumn <- "orig.ident"
}
if(opt$fileType == "seurat" & is.null(opt$clinicalColumns)){
    opt$clinicalColumns <- c("Condition")
}
if(opt$fileType == "seurat" & is.null(opt$clusterColumn)){
    opt$clusterColumn <- "seurat_clusters"
}

# Default columns in scanpy objects if not defined
if(opt$fileType == "scanpy" & is.null(opt$sampleColumn)){
    opt$sampleColumn <- "ind_cov"
}
if(opt$fileType == "scanpy" & is.null(opt$clinicalColumns)){
    opt$clinicalColumns <- c("Status")
}
if(opt$fileType == "scanpy" & is.null(opt$clusterColumn)){
    opt$clusterColumn <- "louvain"
}

# Assign parameters to variables
inputPath <- opt$inputPath
fileType <- opt$fileType
slotSeurat <- opt$slotSeurat
sampleColumn <- opt$sampleColumn
clinicalColumns <- unlist(strsplit(opt$clinicalColumns, ","))
targetColumn <- opt$targetColumn
clusterColumn <- opt$clusterColumn
filterGenes <- opt$filterGenes
organism <- opt$organism
minCellsSample <- opt$minCellsSample
minCellsClass <- opt$minCellsClass
minCells <- opt$minCells
maxCells <- opt$maxCells
outPath <- opt$outPath

# Load and install missing packages
if (fileType == "seurat") {
    if (!require("Seurat", character.only = TRUE)) {
        install.packages("Seurat", dependencies = TRUE)
        library("Seurat", character.only = TRUE)
    }
} else {
    if (!require("reticulate", character.only = TRUE)) {
        install.packages("reticulate", dependencies = TRUE)
        library("reticulate", character.only = TRUE)
    }
}

if (filterGenes) {
    if (!require("BiocManager", character.only = TRUE)) {
        install.packages("BiocManager", dependencies = TRUE)
        library("BiocManager", character.only = TRUE)
    }
    if (!require("biomaRt", character.only = TRUE)) {
        BiocManager::install("biomaRt", dependencies = TRUE)
        library("biomaRt", character.only = TRUE)
    }
}

# Set the python path
if(!is.null(opt$pythonPath) & fileType == "scanpy"){
    use_python(opt$pythonPath)
}

# Data loading ------------------------------------------------------------

# Save the expression data in exprMatrix and the cells metadata in metadataCells
if (fileType == "seurat") {
    seuratobj <- readRDS(inputPath)
    exprMatrix <- GetAssayData(seuratobj, slot = slotSeurat)
    metadataCells <- seuratobj[[]]
} else {
    sc = import("scanpy")
    adata = sc$read_h5ad(inputPath)
    exprMatrix <- adata$X
    colnames(exprMatrix) <- rownames(adata$var)
    rownames(exprMatrix) <- rownames(adata$obs)
    metadataCells <- adata$obs
}


# Filtering ---------------------------------------------------------------

# Get the names of samples and clusters/cell types
samples <- unique(metadataCells[,sampleColumn])
clusters <- unique(metadataCells[,clusterColumn])

# Get the metadata object for samples (metadataSamples). This is done differently
# depending on the number of columns retrieved (one or more than one)
if (length(clinicalColumns) > 1) {
    x <- lapply(samples, function(x) {return(metadataCells[metadataCells[,sampleColumn] == x, clinicalColumns][1,])})
    metadataSamples <- do.call("rbind", x)
    rownames(metadataSamples) <- samples
    metadataSamples$Sample <- samples
    colnames(metadataSamples) <- c(clinicalColumns, "Sample")
} else {
    x <- lapply(samples, function(x) {return(as.character(metadataCells[metadataCells[,sampleColumn] == x, clinicalColumns][1]))})
    metadataSamples <- data.frame(do.call("rbind", x))
    rownames(metadataSamples) <- samples
    metadataSamples$Sample <- samples
    colnames(metadataSamples) <- c(clinicalColumns, "Sample")
}

# Calculate the number of samples with the specified minimum of cells in each cluster
clusterSamples <- c()
clusterClassCells <- c()
clusterNCells <- c()
for (cluster in clusters) {
    metaCluster <- metadataCells[metadataCells[,clusterColumn] == cluster,]
    nCellsCluster <- sapply(samples, function(x) {sum(metaCluster[,sampleColumn] == x)})
    nSamplesPass <- sum(nCellsCluster >= minCellsSample)
    clusterSamples <- c(clusterSamples, nSamplesPass)
    clusterClassCells <- c(clusterClassCells, min(table(metaCluster[,targetColumn])))
    clusterNCells <- c(clusterNCells, nrow(metaCluster))
}
names(clusterSamples) <- clusters
names(clusterClassCells) <- clusters


# Gene filtering ----------------------------------------------------------
# Get the gene names from the expression data
if (fileType == "seurat") {
    genesData <- rownames(exprMatrix)
} else {
    genesData <- colnames(exprMatrix)
}

if (filterGenes) {
    
    # Get gene annotation
    dataset <- paste0(organism, "_gene_ensembl")
    mart <- useMart("ENSEMBL_MART_ENSEMBL", dataset = dataset)
    annot <- getBM(c("external_gene_name","gene_biotype"), mart = mart,
                   filters = "external_gene_name", values = genesData)
    
    # Find mitochondrial, ribosomal, hemoglobin and non-coding genes
    mitGenes <- grep("^MT-", toupper(genesData), value = T)
    ribGenes <- grep("^RPL", toupper(genesData), value = T)
    ribGenes <- c(ribGenes, grep("^RPS", toupper(genesData), value = T))
    hbGenes <- grep("^HB", toupper(genesData), value = T)
    nonCodingGenes <- annot[annot[,2] != "protein_coding",1]
    listExclusion <- c(mitGenes, ribGenes, hbGenes, nonCodingGenes)
    
    # Discard the previous genes from the expression data
    if (fileType == "seurat") {
        exprMatrix <- exprMatrix[!genesData %in% listExclusion,]
    } else {
        exprMatrix <- exprMatrix[,!genesData %in% listExclusion]
    }
    
}

# Save data for singleDeep ------------------------------------------------

# Define the clusters that pass the stablished thresholds
clustersOK <- names(clusterSamples)[clusterSamples == length(samples) & 
                                      clusterClassCells >= minCellsClass & 
                                      clusterNCells >= minCells]

dir.create(outPath, showWarnings = FALSE)
set.seed(123)

# Subsample cells if necessary and save the gene expression and metadata tables
for (cluster in clustersOK) {
    metaCluster <- metadataCells[metadataCells[,clusterColumn] == cluster,]
    cellsCluster <- rownames(metaCluster)
    # Cells subsampling
    if (length(cellsCluster) > maxCells) {
        cellsCluster <- sample(cellsCluster, maxCells, replace = F)
        metaCluster <- metaCluster[cellsCluster,]
    }
    if(fileType == "seurat"){
        exprCluster <- exprMatrix[,cellsCluster]
    }
    else{
        exprCluster <- as.matrix(exprMatrix[cellsCluster,])
        exprCluster <- t(exprCluster)
    }

    cluster <- gsub("/", "_", cluster) # To avoid path problems
    write.table(exprCluster, paste0(outPath, "/", cluster, ".tsv"), sep="\t", quote = F)
    colnames(metaCluster) <- gsub(" ","_", colnames(metaCluster))
    write.table(metaCluster, paste0(outPath, "/", "Metadata_", cluster, ".tsv"), sep="\t", quote = F)
}

colnames(metadataSamples) <- gsub(" ","_", colnames(metadataSamples))
write.table(metadataSamples, paste0(outPath, "/Phenodata.tsv"), sep="\t", quote = F)

# Save the file with the gene names
if(fileType == "seurat"){
    write.table(rownames(exprMatrix), paste0(outPath, "/genes.txt"), sep="\t", quote = F)
} else{
    write.table(colnames(exprMatrix), paste0(outPath, "/genes.txt"), sep="\t", quote = F)
}
