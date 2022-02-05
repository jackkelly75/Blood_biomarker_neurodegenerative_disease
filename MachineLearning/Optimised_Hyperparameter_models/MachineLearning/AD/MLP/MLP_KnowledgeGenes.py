#####################
#	import packages
#####################
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold
from sklearn import metrics
import numpy 
import pylab as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc, precision_recall_curve
from skopt import BayesSearchCV
from matplotlib import pyplot
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

#####################
#	import the data
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\3_scaled+sorted_knowledge\\AD HC')
X_train = pd.read_csv('GSE63061.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('gse63061_pData.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('GSE63060.txt', sep="\t", header= 0, index_col=0)
y_test = pd.read_csv('gse63060_pData.txt', sep="\t", header= 0, index_col=0)

X = X_train.values

y_train = y_train["status"].values.ravel()
y_test = y_test["status"].values.ravel()

for n, i in enumerate(y_train):
    if i == 2:
        y_train[n] = 1

for n, i in enumerate(y_test):
    if i == 2:
        y_test[n] = 1

def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))


#####################
#   Tune the model
#####################
bayes_cv_tuner = GridSearchCV(
    estimator = MLPClassifier(random_state=10, max_iter = 1000),
    param_grid = {
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001],
        'hidden_layer_sizes': [(100,)],
        'solver': ['adam', 'lbfgs', 'sgd']
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 2,
    verbose = 2,
    refit = True,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train)
#0.7347321715086238
#{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'solver': 'lbfgs'}


bayes_cv_tuner = GridSearchCV(
    estimator = MLPClassifier(random_state=10, max_iter = 1000),
    param_grid = {
        'activation': ['relu'],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 1, 10],
        'hidden_layer_sizes': [(10,), (10,10), (100, ), (100,100)],
        'solver': ['lbfgs']
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 2,
    verbose = 10,
    refit = True,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train)

#{'activation': 'relu', 'alpha': 1, 'hidden_layer_sizes': (10, 10), 'solver': 'lbfgs'}
#0.7523810909068701


bayes_cv_tuner = GridSearchCV(
    estimator = MLPClassifier(random_state=10, max_iter = 1000),
    param_grid = {
        'activation': ['relu'],
        'alpha': [0.1, 0.5, 1, 5, 10 ],
        'hidden_layer_sizes': [(5,5),(10,10), (15,15 )],
        'solver': ['lbfgs']
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 2,
    verbose = 10,
    refit = True,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train)
#0.7662261771907979
#{'activation': 'relu', 'alpha': 5, 'hidden_layer_sizes': (15, 15), 'solver': 'lbfgs'}


bayes_cv_tuner = GridSearchCV(
    estimator = MLPClassifier(random_state=10, max_iter = 1000),
    param_grid = {
        'activation': ['relu'],
        'alpha': [3, 4, 5, 6, 7 ],
        'hidden_layer_sizes': [(15,15 ), (20,20), (25,25)],
        'solver': ['lbfgs']
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 2,
    verbose = 10,
    refit = True,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train)
#{'activation': 'relu', 'alpha': 5, 'hidden_layer_sizes': (15, 15), 'solver': 'lbfgs'}
#0.7662261771907979


#####################
#   Run the model
#####################
clf = MLPClassifier(alpha = 5, activation = 'relu', hidden_layer_sizes=(15,15 ), solver = 'lbfgs', verbose=10,  random_state=10, max_iter = 1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.7206477732793523
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.684     0.625     0.653       104
#           1      0.743     0.790     0.766       143
#    accuracy                          0.721       247
#   macro avg      0.714     0.708     0.710       247
#weighted avg      0.718     0.721     0.719       247


#####################
#   Plot the model evaluation
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\5_ML\\AD HC\\5_MLP')

probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.8031199569661108
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('MLP_ROC_bayesian_KnowledgeGenes.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.8293260285742491
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('MLP_prAUC_bayesian_KnowledgeGenes.png', dpi = 2000)
