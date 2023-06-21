from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
from typing import Union, Optional, Tuple, Collection, Sequence, Iterable
import argparse
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import logging as logg
from sklearn.utils import sparsefuncs, check_array
import gc as py_gc

fileout = open("log.log", "w")
fileout.write("Log file\n")
fileout.close()

##############################
# Science Dataset Processing #
##############################
input_file = "GSE174188_CLUES1_adjusted.h5ad"

adata = sc.read_h5ad(input_file)
# adata = adata[adata.obs['ind_cov'].isin(['HC-543', 'HC-551', '1248_1248', '1019_1019']),:]
fileout = open("log.log", "a")
fileout.write("Read GSE174188\n")
fileout.close()


X = adata.raw.X
obs = pd.DataFrame()
obs['Status'] = adata.obs['SLE_status'].tolist()
obs['ind_cov'] = adata.obs['ind_cov'].tolist()
obs['louvain'] = adata.obs['louvain'].tolist()
obs['batch_cov'] = adata.obs['batch_cov'].tolist()
obs['pop_cov'] = adata.obs['pop_cov'].tolist()
obs['cg_cov'] = adata.obs['cg_cov'].tolist()
obs['ct_cov'] = adata.obs['ct_cov'].tolist()
obs['Age'] = adata.obs['Age'].tolist()
obs['Sex'] = adata.obs['Sex'].tolist()
obs['Processing_Cohort'] = adata.obs['Processing_Cohort'].tolist()
obs['L3'] = adata.obs['L3'].tolist()
var_names = adata.raw.var_names.tolist()
var = pd.DataFrame(index=var_names)
cdata = ad.AnnData(X, obs=obs, var=var, dtype='int32')
cdata.obs_names = adata.obs_names
# cdata.raw = cdata

del(adata)
py_gc.collect()
# annotate the group of mitochondrial genes as 'mt'
cdata.var['mt'] = cdata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(
    cdata, qc_vars=['mt'], percent_top=None, log1p=True, inplace=True)

ncells = cdata.shape[0]

sc.pp.filter_cells(cdata, min_genes=100)
sc.pp.filter_cells(cdata, max_genes=4000)
sc.pp.filter_genes(cdata, min_cells=ncells/1000)

fileout = open("log.log", "a")
fileout.write("Filtered GSE174188\n")
fileout.close()


sc.pp.normalize_total(cdata)
sc.pp.log1p(cdata)
py_gc.collect()

fileout = open("log.log", "a")
fileout.write("Starting Combat GSE174188\n")
fileout.close()
sc.pp.combat(cdata, key="batch_cov")
fileout = open("log.log", "a")
fileout.write("End Combat GSE174188\n")
fileout.close()

py_gc.collect()



################################
# pediatric dataset Processing #
################################
input_file = "raw.h5ad"
fileout = open("log.log", "a")
fileout.write("Read pediatric\n")
fileout.close()

adata = sc.read_h5ad(input_file)
# adata = adata[adata.obs['orig.ident'].isin(['GSM4029906', 'GSM4029939', 'GSM4029936', 'GSM4029907']),:]
X = adata.raw.X
obs = pd.DataFrame()
obs['Status'] = adata.obs['Condition'].tolist()
obs['ind_cov'] = adata.obs['orig.ident'].tolist()
obs['batch_cov'] = adata.obs['Batch'].tolist()
var_names = adata.var_names.tolist()
var = pd.DataFrame(index=var_names)
pedata = ad.AnnData(X, obs=obs, var=var, dtype='int32')
pedata.obs_names = adata.obs_names
# pedata.raw = pedata

del(adata)
py_gc.collect()

# annotate the group of mitochondrial genes as 'mt'
pedata.var['mt'] = pedata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(
    pedata, qc_vars=['mt'], percent_top=None, log1p=True, inplace=True)

ncells = pedata.shape[0]

sc.pp.filter_cells(pedata, min_genes=100)
sc.pp.filter_cells(pedata, max_genes=4000)
sc.pp.filter_genes(pedata, min_cells=ncells/1000)

fileout = open("log.log", "a")
fileout.write("Filtered pediatric\n")
fileout.close()

sc.pp.normalize_total(pedata)
sc.pp.log1p(pedata)

fileout = open("log.log", "a")
fileout.write("Starting combat pediatric\n")
fileout.close()
sc.pp.combat(pedata, key="batch_cov")
fileout = open("log.log", "a")
fileout.write("End combat pediatric\n")
fileout.close()


#################
# Concatenation #
#################
fileout = open("log.log", "a")
fileout.write("Merging datasets\n")
fileout.close()

cdata.obs['dataset'] = "science"
pedata.obs['dataset'] = "pediatric"

common_genes = list(set(cdata.var_names) & set(pedata.var_names))
cdata = cdata[:, common_genes]
pedata = pedata[:, common_genes]

mergedData = ad.concat([cdata, pedata], join="outer")

del(cdata)
del(pedata)
py_gc.collect()

#########################
# Regression and COMBAT #
#########################

# fileout = open("log.log", "a")
# fileout.write("Generating UMAP before batch correction\n")
# fileout.close()
# 
initialization = 1
# sc.pp.pca(mergedData, random_state=initialization, svd_solver='arpack')
# sc.pp.neighbors(mergedData, random_state=initialization)
# sc.tl.umap(mergedData, random_state=initialization)
# sc.pl.umap(mergedData, color = 'dataset', title="Batch no corrected", save="BatchNoCorrected.png")

py_gc.collect()
fileout = open("log.log", "a")
fileout.write("Starting combat between datasets\n")
fileout.close()
sc.pp.combat(mergedData, key="dataset")
fileout = open("log.log", "a")
fileout.write("End combat between datasets\n")
fileout.close()
py_gc.collect()

fileout = open("log.log", "a")
fileout.write("Selecting genes\n")
fileout.close()

sc.pp.highly_variable_genes(mergedData, n_top_genes=2000, inplace=True, subset=True, batch_key='dataset')

fileout = open("log.log", "a")
fileout.write("Regressing mt expression\n")
fileout.close()
sc.pp.regress_out(mergedData, ['total_counts', 'pct_counts_mt'])
# X_scaled, mean, std = sc.preprocessing._simple.scale_array(
#     cdata_filt.X, return_mean_std=True, max_value=10)

hv_genes = list(mergedData.var_names)

cdata_filt = mergedData[mergedData.obs['dataset'] == 'science', hv_genes]
pedata_filt = mergedData[mergedData.obs['dataset'] == 'pediatric', hv_genes]

fileout = open("log.log", "a")
fileout.write("Writing science\n")
fileout.close()
cdata_filt.write_h5ad("science_batchCorrected.h5ad")




fileout = open("log.log", "a")
fileout.write("Generating UMAP after batch correction\n")
fileout.close()
sc.pp.pca(mergedData, random_state=initialization, svd_solver='arpack')
sc.pp.neighbors(mergedData, random_state=initialization)
sc.tl.umap(mergedData, random_state=initialization)
sc.pl.umap(mergedData, color = 'dataset', title="Batch corrected", save="BatchCorrected.png")

del(mergedData)
py_gc.collect()

########################
### Cells projection ###
########################
fileout = open("log.log", "a")
fileout.write("Projecting cells\n")
fileout.close()

sc.pp.pca(cdata_filt)
sc.pp.neighbors(cdata_filt)
sc.tl.umap(cdata_filt)

sc.tl.ingest(pedata_filt, cdata_filt, obs='cg_cov')


fileout = open("log.log", "a")
fileout.write("Writing pediatric\n")
fileout.close()
pedata_filt.write_h5ad("pediatric_batchCorrected.h5ad")

fileout = open("log.log", "a")
fileout.write("UMAP projection\n")
fileout.close()
sc.pp.pca(pedata_filt, random_state=initialization, svd_solver='arpack')
sc.pp.neighbors(pedata_filt, random_state=initialization)
sc.tl.umap(pedata_filt, random_state=initialization)
sc.pl.umap(pedata_filt, color = 'cg_cov', title="Pediatric projected", save="PediatricProjected.png")
# def scale_transform_array(X, *, mean, std, zero_center: bool = True, max_value: Optional[float] = None, copy: bool = False, return_mean_std: bool = False):
#     if copy:
#         X = X.copy()
#     if issparse(X):
#         if zero_center:
#             raise ValueError("Cannot zero-center sparse matrix.")
#         sparsefuncs.inplace_column_scale(X, 1 / std)
#     else:
#         if zero_center:
#             X -= mean
#         X /= std
#     if max_value is not None:
#         logg.debug(f"... clipping at max_value {max_value}")
#         X[X > max_value] = max_value
#     if return_mean_std:
#         return X, mean, std
#     else:
#         return X
# 
# 
# pdata_x = scale_transform_array(X=pedata_filt.X, mean=mean, std=std)
# 
# pedata_filt.X = pdata_x
# cdata_filt.X = X_scaled


