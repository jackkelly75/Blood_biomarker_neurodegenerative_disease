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
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\3_scaled+sorted_data')

y_train = pd.read_csv('y_train.txt', sep="\t", header= 0, index_col=0)
y_test = pd.read_csv('y_test.txt', sep="\t", header= 0, index_col=0)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

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



#########
# LASSO
#########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\4_FeatureSelection\\LASSO')

with open("X_train_LASSO", "rb") as input_file:
    X_train_LASSO = pickle.load(input_file)

with open("X_test_LASSO", "rb") as input_file:
    X_test_LASSO = pickle.load(input_file)


############
#	Tune the model
############
bayes_cv_tuner = GridSearchCV(
    estimator = MLPClassifier(random_state=10, max_iter = 10000, tol = 0.00001),
    param_grid = {
        'activation': ['relu'],
        'learning_rate': ['constant'],
        'alpha': [1],
        'hidden_layer_sizes': [(100,)],
        'solver': ['sgd', 'adam', 'lbfgs']
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

result = bayes_cv_tuner.fit(X_train_LASSO, y_train)
#{'activation': 'relu', 'alpha': 1, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'solver': 'sgd'}


bayes_cv_tuner = GridSearchCV(
    estimator = MLPClassifier(random_state=10, max_iter = 10000, tol = 0.00001, solver = 'sgd'),
    param_grid = {
        'activation': ['relu'],
        'learning_rate': ['constant','adaptive'],
        'alpha': [0.01, 0.1, 1, 100],
        'hidden_layer_sizes': [(10,), (100,), (10,10), (100,100)],
        'solver': ['sgd']
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

result = bayes_cv_tuner.fit(X_train_LASSO, y_train)
#{'activation': 'relu', 'alpha': 1, 'hidden_layer_sizes': (10,), 'learning_rate': 'adaptive', 'solver': 'sgd'}
#0.7924148101106846

bayes_cv_tuner = GridSearchCV(
    estimator = MLPClassifier(random_state=10, max_iter = 10000, tol = 0.00001, solver = 'sgd'),
    param_grid = {
        'activation': ['relu'],
        'learning_rate': ['adaptive'],
        'alpha': [0.5, 1, 5, 10],
        'hidden_layer_sizes': [(15,), (10, ), (20, ), (5,)]
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

result = bayes_cv_tuner.fit(X_train_LASSO, y_train)
#{'activation': 'relu', 'alpha': 5, 'hidden_layer_sizes': (5,), 'learning_rate': 'adaptive'}
#0.8071193022273837

bayes_cv_tuner = GridSearchCV(
    estimator = MLPClassifier(random_state=10, max_iter = 10000, tol = 0.00001, solver = 'sgd'),
    param_grid = {
        'activation': ['relu'],
        'learning_rate': ['adaptive'],
        'alpha': [2, 3, 4, 5, 6, 7, 8],
        'hidden_layer_sizes': [(5,), (2,), (3, ), (4,), (6,), (7,)]
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

result = bayes_cv_tuner.fit(X_train_LASSO, y_train)
#{'activation': 'relu', 'alpha': 3, 'hidden_layer_sizes': (5,), 'learning_rate': 'adaptive'}
# 0.8074558132680909


############
#	Run the model
############
clf = MLPClassifier(activation = 'relu', hidden_layer_sizes=(5,), max_iter=10000, alpha=3,solver='sgd', learning_rate = 'adaptive', verbose=10,  random_state=10, tol=0.00001)
clf.fit(X_train_LASSO, y_train)
y_pred = clf.predict(X_test_LASSO)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.6259541984732825
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.627     0.691     0.657        68
#           1      0.625     0.556     0.588        63
#    accuracy                          0.626       131
#   macro avg      0.626     0.623     0.623       131
#weighted avg      0.626     0.626     0.624       131


############
#	Plot the model evaluation
############
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\5_ML\\6_MLP')

probas_ = clf.fit(X_train_LASSO, y_train).predict_proba(X_test_LASSO)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.6631652661064426
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('MLP_ROC_bayesian_LASSO.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.6255929688447477
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('MLP_prAUC_bayesian_LASSO.png', dpi = 2000)

