#####################
#	import packages for data
#####################
import pandas as pd
import os
import numpy 
import scipy
import scipy.stats

#####################
#	set wd
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\2 prepared data\\AD HC')

#####################
#	import data
#####################
GSE63060 = pd.read_csv('GSE63060.txt', sep="\t", header= 0)
gse63060_pData = pd.read_csv('gse63060_pData.txt', sep="\t", header= 0)
# features
GSE63061 = pd.read_csv('GSE63061.txt', sep="\t", header= 0)
gse63061_pData = pd.read_csv('gse63061_pData.txt', sep="\t", header= 0)

#####################
#	keep same features
#####################
common = set(list(GSE63061.columns.values)) & set(list(GSE63060.columns.values))
GSE63061 = pd.DataFrame(GSE63061,columns=common)
GSE63060 = pd.DataFrame(GSE63060,columns=common)

#####################
#	Find top 3000 MAD genes
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\3_scaled+sorted_knowledge\\AD HC')
x = scipy.stats.median_absolute_deviation(GSE63061)
x = GSE63061[GSE63061.columns[x.argsort()[:3000]]]
ino = pd.DataFrame(x.columns.values)
MAD_genes = ino[0].tolist()
#export to csv
ino.to_csv('top_3000_MAD.txt', sep="\t", index = False)

#####################
#	Keep knowledge based genes
#####################
knowledge = pd.read_csv('prev_features.txt', sep="\t")
prev_knowledge = knowledge['prev_genes'].tolist()
prev_knowledge = list(set(prev_knowledge))

gene_list = MAD_genes + list(set(prev_knowledge) - set(MAD_genes))

GSE63061 = GSE63061[GSE63061.columns.intersection(gene_list)]
GSE63060 = GSE63060[GSE63060.columns.intersection(gene_list)]
#7520 genes


#####################
#	Convert age to features
#####################
GSE63061['age']= gse63061_pData['age']
GSE63060['age']= gse63060_pData['age']


#####################
#	Change status to numerics
#####################
status = {'CTL': 0, 'MCI': 1, 'AD': 2}
#Assign these different key-value pair from above dictiionary to your table
gse63060_pData.status = [status[item] for item in gse63060_pData.status]
gse63061_pData.status = [status[item] for item in gse63061_pData.status]

######################
#	Standardize data
######################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
GSE63061[GSE63061.columns] = scaler.fit_transform(GSE63061[GSE63061.columns])
GSE63060[GSE63060.columns] = scaler.fit_transform(GSE63060[GSE63060.columns])

#####################
#	save results
#####################
GSE63061.to_csv('GSE63061.txt', sep="\t")
gse63061_pData.to_csv('gse63061_pData.txt', sep="\t")
GSE63060.to_csv('GSE63060.txt', sep="\t")
gse63060_pData.to_csv('gse63060_pData.txt', sep="\t")

