#####################
#	import packages for data
#####################
import pandas as pd
import os
import numpy 

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
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\3_scaled+sorted_data\\AD HC')
GSE63061.to_csv('GSE63061.txt', sep="\t")
gse63061_pData.to_csv('gse63061_pData.txt', sep="\t")
GSE63060.to_csv('GSE63060.txt', sep="\t")
gse63060_pData.to_csv('gse63060_pData.txt', sep="\t")
