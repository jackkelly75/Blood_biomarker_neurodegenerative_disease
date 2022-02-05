#######
# T-sne
#######
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from bioinfokit.visuz import cluster
import os
from sklearn.preprocessing import StandardScaler

os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\3_scaled+sorted_data\\AD HC')

########
#for test+training data
#######
GSE63060 = pd.read_csv('GSE63060.txt', sep="\t", header= 0, index_col=0)
gse63060_pData = pd.read_csv('gse63060_pData.txt', sep="\t", header= 0, index_col=0)

scaler = StandardScaler() 
GSE63060[GSE63060.columns] = scaler.fit_transform(GSE63060[GSE63060.columns])

os.chdir('C:\\Users\\Jack\\Desktop')
tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(GSE63060)

y = gse63060_pData['status']
color_class = np.where(y.to_numpy() > 0.5, "Alzheimer's", "Control")
cluster.tsneplot(score=tsne_em, colorlist=color_class, legendpos='upper right', legendanchor=(0.3, 1), dim = (6,6), r =2000 , figname = "t-SNE_status_all", colordot = ("#003f5c", "#ffa600"))


##################### other dataset


os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\3_scaled+sorted_data\\AD HC')

########
#for test+training data
#######
GSE63061 = pd.read_csv('GSE63061.txt', sep="\t", header= 0, index_col=0)
gse63061_pData = pd.read_csv('gse63061_pData.txt', sep="\t", header= 0, index_col=0)

scaler = StandardScaler() 
GSE63061[GSE63061.columns] = scaler.fit_transform(GSE63061[GSE63061.columns])

os.chdir('C:\\Users\\Jack\\Desktop')
tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(GSE63061)


y = gse63061_pData['status']
color_class = np.where(y.to_numpy() > 0.5, "Alzheimer's", "Control")
cluster.tsneplot(score=tsne_em, colorlist=color_class, legendpos='upper right', legendanchor=(0.3, 1), dim = (6,6), r =2000 , figname = "t-SNE_status_all", colordot = ("#003f5c", "#ffa600"))
