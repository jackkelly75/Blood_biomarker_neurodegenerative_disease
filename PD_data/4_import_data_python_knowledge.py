#####################
#	import packages for data
#####################
import pandas as pd
import os
import numpy 
import scipy
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#####################
#	set wd
#####################
os.chdir('/home/jack11/Desktop/PD_ML_final/2_prepared_data')

#####################
#	import data
#####################
GSE99039 = pd.read_csv('GSE99039.txt', sep="\t", header= 0)
gse99039_pData = pd.read_csv('gse99039_pData.txt', sep="\t", header= 0)

#####################
#	Find top 3000 MAD genes
#####################
os.chdir('/home/jack11/Desktop/PD_ML_final/3_scaled+sorted_knowledge')
x = scipy.stats.median_absolute_deviation(GSE99039)
x = GSE99039[GSE99039.columns[x.argsort()[:3000]]]
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
GSE99039 = GSE99039[GSE99039.columns.intersection(gene_list)]
#4981 features

#####################
#	Change status to numerics
#####################
status = {'CONTROL': 0, 'IPD': 1}
#Assign these different key-value pair from above dictionary to your table
gse99039_pData['disease label:ch1'] = [status[item] for item in gse99039_pData['disease label:ch1']]

#####################
#	save results
#####################
GSE99039.to_csv('GSE99039.txt', sep="\t")
gse99039_pData.to_csv('gse99039_pData.txt', sep="\t")

#####################
# Split and save data
######################
y = gse99039_pData['disease label:ch1'] # define the target variable (dependent variable) as y
X_train, X_test, y_train, y_test = train_test_split(GSE99039, y, test_size=0.3, random_state = 12)

scaled_features = StandardScaler().fit_transform(X_train.values)
X_train = pd.DataFrame(scaled_features, index=X_train.index, columns=X_train.columns)

scaled_features = StandardScaler().fit_transform(X_test.values)
X_test = pd.DataFrame(scaled_features, index=X_test.index, columns=X_test.columns)


X_train.to_csv('X_train.txt', sep="\t")
X_test.to_csv('X_test.txt', sep="\t")
y_train.to_csv('y_train.txt', sep="\t")
y_test.to_csv('y_test.txt', sep="\t")
