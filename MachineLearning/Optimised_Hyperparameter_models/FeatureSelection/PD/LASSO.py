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
from skopt import BayesSearchCV
from sklearn.linear_model import Lasso

#####################
#   import data
#####################
#	set wd and import file
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\3_scaled+sorted_data')
X_train = pd.read_csv('X_train.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('X_test.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('y_train.txt', sep="\t", header= 0, index_col=0)
y_train = y_train.values.ravel()
y_test = pd.read_csv('y_test.txt', sep="\t", header= 0, index_col=0)
y_test = y_test.values.ravel()



################################
# Tune LASSO model using bayesian optimisation
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
    estimator = Lasso(random_state= 1, tol = 1e-4, max_iter = 3000),
    search_spaces = {
        'alpha': (1e-3, 1e+02, 'log-uniform')
    },
    scoring = 'average_precision',    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=25
    ),
    n_jobs = 4,
    n_iter = 200,   
    verbose = 0,
    refit = True,
    random_state = 1
)

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
print("LASSO running")

#Model #200
#Best pr-AUC: 0.5861
#Best params: OrderedDict([('alpha', 0.07552181227576159)])


################################
# Use SelectFromModel to get the features
################################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\4_FeatureSelection\\LASSO')

sel_ = SelectFromModel(Lasso(random_state= 1, tol = 1e-4, max_iter = 3000, alpha = 0.07552181227576159))
sel_.fit(X_train, y_train)

# make a list with the selected features and print the outputs
selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))

#selected features: 19


################################
# Save results
################################

#save as text file
with open('LASSO_model.txt', 'w') as f:
    for item in selected_feat:
        f.write("%s\n" % item)

with open('LASSO_model', 'wb') as fp:
    pickle.dump(selected_feat, fp)


X_train_LASSO = X_train[X_train.columns.intersection(selected_feat)]
X_test_LASSO = X_test[X_test.columns.intersection(selected_feat)]

with open('X_train_LASSO', 'wb') as fp:
    pickle.dump(X_train_LASSO, fp)

with open('X_test_LASSO', 'wb') as fp:
    pickle.dump(X_test_LASSO, fp)
