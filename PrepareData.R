# Load ans install missing packages
packages = c("optparse", "Seurat", "reticulate")
package.check <- lapply(
    packages,
    FUN = function(x) {
        if (!require(x, character.only = TRUE)) {
            install.packages(x, dependencies = TRUE)
            library(x, character.only = TRUE)
        }
    }
)

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
    make_option(c("--clusterColumn"), type="character", default= NULL,
        help="Specify the columns with cluster information"),
    make_option(c("--minCells"), type="numeric", default= 0,
        help="Minimum number of cells that must contain 
        each sample in each cluster"),
    make_option(c("--maxCells"), type="numeric", default= 50000,
        help="Maximum number of cells per cluster. If the number of 
        cells is higher than this, random cells are removed"),
    make_option(c("--outPath"), type="character", default= "outData",
        help="Output path")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

#Controls:
#opt$fileType
if(!(opt$fileType== "scanpy" | opt$fileType == "seurat")){
    stop("It has not been specified if the file type is scanpy or seurat")
}

#Default slotSeurat
if(opt$fileType == "seurat" & is.null(opt$slotSeurat)){
    opt$slotSeurat <- "scale.data"
}

#Default columns in Seurat objects
# sample
if(opt$fileType == "seurat" & is.null(opt$sampleColumn)){
    opt$sampleColumn <- "orig.ident"
}
#clinical
if(opt$fileType == "seurat" & is.null(opt$clinicalColumns)){
    opt$clinicalColumns <- c("Condition")
}
#cluster
if(opt$fileType == "seurat" & is.null(opt$clusterColumn)){
    opt$clusterColumn <- "seurat_clusters"
}

#Default columns in scanpy objects
# sample
if(opt$fileType == "scanpy" & is.null(opt$sampleColumn)){
    opt$sampleColumn <- "ind_cov"
}
#clinical
if(opt$fileType == "scanpy" & is.null(opt$clinicalColumns)){
    opt$clinicalColumns <- c("Status")
}
#cluster
if(opt$fileType == "scanpy" & is.null(opt$clusterColumn)){
    opt$clusterColumn <- "louvain"
}

inputPath <- opt$inputPath
fileType <- opt$fileType
slotSeurat <- opt$slotSeurat 
sampleColumn <- opt$sampleColumn
clinicalColumns <- unlist(strsplit(opt$clinicalColumns, ","))
clusterColumn <- opt$clusterColumn
minCells <- opt$minCells
maxCells <- opt$maxCells
outPath <- opt$outPath



# Data loading ------------------------------------------------------------
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

samples <- unique(metadataCells[,sampleColumn])
clusters <- unique(metadataCells[,clusterColumn])

if (length(clinicalColumns) > 1) {
    x <- lapply(samples, function(x) {return(metadataCells[metadataCells[,sampleColumn] == x, clinicalColumns][1,])})
    metadataSamples <- do.call("rbind", x)
    rownames(metadataSamples) <- samples
    metadataSamples$Sample <- samples
    colnames(metadataSamples) <- c(clinicalColumns, "Sample")
} else {
    x <- lapply(samples, function(x) {return(metadataCells[metadataCells[,sampleColumn] == x, clinicalColumns][1])})
    metadataSamples <- data.frame(do.call("rbind", x))
    rownames(metadataSamples) <- samples
    metadataSamples$Sample <- samples
    colnames(metadataSamples) <- c(clinicalColumns, "Sample")
}

# Calculate the number of samples with a minimum of cells in each cluster
clusterSamples <- c()
for (cluster in clusters) {
    metaCluster <- metadataCells[metadataCells[,clusterColumn] == cluster,]
    nCellsCluster <- sapply(samples, function(x) {sum(metaCluster[,sampleColumn] == x)})
    nSamplesPass <- sum(nCellsCluster >= minCells)
    clusterSamples <- c(clusterSamples, nSamplesPass)
}
names(clusterSamples) <- clusters

# Save data for singleDeep ------------------------------------------------

clustersOK <- names(clusterSamples)[clusterSamples == length(samples)]

dir.create(outPath, showWarnings = FALSE)
set.seed(123)
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
        exprCluster <- exprMatrix[cellsCluster,]
        exprCluster <- t(exprCluster)
    }
    
    write.table(exprCluster, paste0(outPath, "/", cluster, ".tsv"), sep="\t", quote = F)
    write.table(metaCluster, paste0(outPath, "/", "Metadata_", cluster, ".tsv"), sep="\t", quote = F)
}


write.table(metadataSamples, paste0(outPath, "/Phenodata.tsv"), sep="\t", quote = F)

if(fileType == "seurat"){
    write.table(rownames(exprMatrix), paste0(outPath, "/genes.txt"), sep="\t", quote = F)
} else{
    write.table(colnames(exprMatrix), paste0(outPath, "/genes.txt"), sep="\t", quote = F)
}




