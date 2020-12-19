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
from matplotlib import pyplot
from sklearn.metrics import precision_score, recall_score, accuracy_score, precision_recall_curve, roc_curve, auc
from numpy import mean
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from skopt import BayesSearchCV


#####################
#   import data
#####################
#   set wd and import file
X_train = pd.read_csv('X_train.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('X_test.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('y_train.txt', sep="\t", header= 0, index_col=0)
y_train = y_train.values.ravel()
y_test = pd.read_csv('y_test.txt', sep="\t", header= 0, index_col=0)
y_test = y_test.values.ravel()


################################
# ElasticNet
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
    estimator = ElasticNet(random_state=1,  max_iter = 5000),
    search_spaces = {
        'alpha': (1e-4, 10.0, 'log-uniform'),
        'l1_ratio': (0.001, 1.0, 'uniform')
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
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)

#replace alpha and l1_ratio with optimum found using BayesSearchCV()
sel_ = SelectFromModel(ElasticNet(random_state=1, max_iter = 5000, alpha = , l1_ratio = ))
sel_.fit(X_train, y_train)

# make a list with the selected features and print the outputs
selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))


#save model as text and pickle file
with open('ElasticNet_model.txt', 'w') as f:
    for item in selected_feat:
        f.write("%s\n" % item)

with open('ElasticNet_model', 'wb') as fp:
    pickle.dump(selected_feat, fp)


#save X_train and X_test with only features found
X_train_ElasticNet = X_train[X_train.columns.intersection(selected_feat)]
X_test_ElasticNet = X_test[X_test.columns.intersection(selected_feat)]

with open('X_train_ElasticNet', 'wb') as fp:
    pickle.dump(X_train_ElasticNet, fp)

with open('X_test_ElasticNet', 'wb') as fp:
    pickle.dump(X_test_ElasticNet, fp)
