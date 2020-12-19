#####################
#	import packages for feature selection
#####################
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import pickle
import numpy as np
from sklearn import metrics
from numpy import mean
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RepeatedKFold
from skopt import BayesSearchCV
from sklearn.utils import class_weight
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, precision_recall_curve, roc_curve, auc

#####################
#	import data
#####################
X_train = pd.read_csv('X_train.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('X_test.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('y_train.txt', sep="\t", header= 0, index_col=0)
y_train = y_train.values.ravel()
y_test = pd.read_csv('y_test.txt', sep="\t", header= 0, index_col=0)
y_test = y_test.values.ravel()


################################
# Tune LogisticRegression
################################

def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest pr-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

bayes_cv_tuner = BayesSearchCV(
    estimator = LogisticRegression(random_state=1,  class_weight='balanced', penalty='l2', solver='liblinear'),
    search_spaces = {
        'C': (1e-7, 1e+1, 'log-uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=25
    ),
    n_jobs = 2,
    n_iter = 200,
    verbose = 0,
    refit = True,
    random_state = 1,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)



###########################
# VSSRFE
###########################
# define the C_value with the optimum C parameter
C_value = 
data = X_train
label = y_train
#max iter reduced as will converge with this model
def VSSRFE(n_genes):
    n_selected = n_genes
    X=data
    Y=label
    s_initial = 100
    m, n_total = X.shape
    temp = n_total
    N = n_total
    S = s_initial
    clf = LogisticRegression(C= C_value, random_state=1,  class_weight='balanced', penalty='l2', solver='liblinear')
    while N > n_selected:
        N = N-S
        if temp/N>2 and S>1:
            temp=N
            S=S/2
            S=round(S)
        clf.fit(X, Y)
        coef=[abs(e) for e in (clf.coef_)[0]]
        coef_sorted=np.argsort(coef)[::-1]
        coef_eliminated=coef_sorted[:N]
        X = X.iloc[:, coef_eliminated]
    return X

def model(n_genes):
    print(n_genes)
    data=VSSRFE(n_genes)
    ls_acc=[];ls_auc=[];pr_auc=[]
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=11)
    clf = LogisticRegression(C= C_value, random_state=1,  class_weight='balanced', penalty='l2', solver='liblinear')
    for train_index, test_index in skf.split(data,label):
        clf.fit(data.iloc[train_index], label[train_index])
        acc=clf.score(data.iloc[test_index],label[test_index])
        ls_acc.append(acc)
        fpr, tpr, thresholds = metrics.roc_curve(label[test_index], (clf.predict_proba(data.iloc[test_index]))[:,1])
        auc=metrics.auc(fpr,tpr)
        ls_auc.append(auc)
        precision, recall, _ = metrics.precision_recall_curve(label[test_index], (clf.predict_proba(data.iloc[test_index]))[:,1])
        prauc = metrics.auc(recall, precision)
        pr_auc.append(prauc)
    return np.mean(ls_acc), np.mean(ls_auc), np.mean(pr_auc)

start = time.time()
res_acc=[];res_auc=[];res_prAUC=[]
for i in range(1, 31):
    ac,uc,pruc=model(n_genes = i)
    res_acc.append(ac)
    res_auc.append(uc)
    res_prAUC.append(pruc)

end = time.time()
print(end - start)
print("VSSRFE+LR")


#save the list of accuracy, ROCAUC and prAUC as text files
with open('res_acc.txt', 'w') as f:
    for item in res_acc:
        f.write("%s\n" % item)

with open('res_auc.txt', 'w') as f:
    for item in res_auc:
        f.write("%s\n" % item)

with open('res_prAUC.txt', 'w') as f:
    for item in res_prAUC:
        f.write("%s\n" % item)


#plot them
myTitle = 'Comparison of performace score obtained using different number of genes found by VSSRFE'
pyplot.figure()
pyplot.title(myTitle, loc='center', wrap=True)
pyplot.xlabel('Number of genes selected') 
pyplot.ylabel('Score')
pyplot.plot(res_prAUC, color="orange", label = "prAUC")
pyplot.plot(res_auc, color="slateblue", label = "ROC_AUC")
pyplot.plot(res_acc, color="green", label = "Accuracy")
pyplot.legend()
pyplot.savefig('VSSRFE_LR_200.png', dpi = 2000)
pyplot.clf()


##################################
#   Get gene list for optimum
##################################
#choose max accuracy as the prAUC and ROCAUC were levelling off at this point
m = max(res_prAUC)
[i for i, j in enumerate(res_prAUC) if j == m]
#add 1 for the number found above to get number of genes (as python prints from 0)
data=VSSRFE(5)

new_features = list(data.columns) 

#save as text file
with open('VSSRFE_LR.txt', 'w') as f:
    for item in new_features:
        f.write("%s\n" % item)

#save as pickle
with open('VSSRFE_LR', 'wb') as fp:
    pickle.dump(new_features, fp)

#save the dataset with only selected features as pickle
X_train_VSSRFE_LR = X_train[X_train.columns.intersection(new_features)]
X_test_VSSRFE_LR = X_test[X_test.columns.intersection(new_features)]

with open('X_train_VSSRFE_LR', 'wb') as fp:
    pickle.dump(X_train_VSSRFE_LR, fp)

with open('X_test_VSSRFE_LR', 'wb') as fp:
    pickle.dump(X_test_VSSRFE_LR, fp)




