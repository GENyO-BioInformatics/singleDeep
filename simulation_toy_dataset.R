# Load and install missing packages ---------------------------------------
packages <- c("devtools", "Seurat")
package.check <- lapply(
    packages,
    FUN = function(x) {
        if (!require(x, character.only = TRUE)) {
            install.packages(x, dependencies = TRUE)
            library(x, character.only = TRUE)
        }
    }
)

if (!require("splatter", character.only = TRUE)) {
    devtools::install_github("Oshlack/splatter")
}
library(splatter)
source("cell_types_performance/custom_splatter_functions.R")


# Simulate data  ----------------------------------------
dir.create("./toy_dataset")
fileName <- "data"
sample_heterogeneity = 0.1
nSamples <- 20
nGroups <- 5

# Groups ore the cell types
# Random cell types proportions
set.seed(123)
prop <- runif(nGroups)
prop = prop/(sum(prop))

params <- newSplatPopParams(nGenes = 1000,
                            batchCells=100,
                            batch.size=nSamples,
                            condition.prob=c(0.5, 0.5),
                            cde.prob = 0.025,
                            cde.facLoc = 0.05,
                            group.prob = prop,
                            similarity.scale=1000, 
                            de.prob = 0.8, 
                            de.facLoc = 0.05,
                            seed=123)

# Reduces the differential expression level for each group
cde.groupSpecific = list(Group1 = 1, Group2 = 0.9, Group3 = 0.8, Group4 = 0.7, Group5 = 0.6)

sim.means <- splatPopSimulateMeans(params,
                                   vcf=mockVCF(n.samples=nSamples),
                                   cde.groupSpecific = cde.groupSpecific,
                                   sample_heterogeneity = sample_heterogeneity)
sim <- splatPopSimulateSC(sim.means = sim.means$means,
                          params = params, key = sim.means$key,
                          conditions = sim.means$conditions, counts.only = F,
                          method = "groups", sparsify = FALSE, verbose = T)


# Save Seurat object
colnames(sim@assays@data@listData$counts) <- paste0(colData(sim)$Sample, "_", colData(sim)$Cell, "_", colData(sim)$Group)
rownames(colData(sim)) = colnames(sim@assays@data@listData$counts)
rownames(sim@assays@data@listData$counts) <- gsub("_", "", rownames(sim@assays@data@listData$counts))

seuratobj = CreateSeuratObject(counts = sim@assays@data@listData$counts,
                               project = "SCSimulation",
                               meta.data = as.data.frame(colData(sim))[,c(3,5,6)],
                               names.field = 1,
                               names.delim = "NA")
seuratobj <- NormalizeData(seuratobj, normalization.method = "LogNormalize")
seuratobj <- ScaleData(seuratobj)
seuratobj <- FindVariableFeatures(seuratobj, nfeatures = 2000)
saveRDS(seuratobj, file = "./toy_dataset/toy.rds")

commandPrepare <- "Rscript singleDeep/PrepareData.R --inputPath toy_dataset/toy.rds --fileType seurat --sampleColumn Sample --clusterColumn Group --clinicalColumns Condition --targetColumn Condition --outPath toy_dataset/data"

system(commandPrepare)
