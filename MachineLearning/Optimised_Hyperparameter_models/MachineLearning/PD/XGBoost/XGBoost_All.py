#####################
#	import packages
#####################
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold, cross_val_score
from sklearn import metrics
import numpy 
import pylab as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc, precision_recall_curve
from skopt import BayesSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from matplotlib import pyplot

#####################
#	import the data
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\3_scaled+sorted_data')
X_train = pd.read_csv('X_train.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('X_test.txt', sep="\t", header= 0, index_col=0)
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
    print('Model #{}\nBest pr-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

(unique, counts) = numpy.unique(y_train, return_counts=True)
frequencies = numpy.asarray((unique, counts)).T
#[  0, 162]
#[  1, 141]
162/141
#1.148936170212766


####################
# optimse model params using gridSearchCV
####################
#Step 1     -   High learning rate (0.1) and find best n_estimators (number of trees)
#Step 2     -   Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) for decided learning rate and number of trees
#Step 3     -   Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
#Step 4     -   Lower learning rate and retune n_estimators


##Step 1
xgtrain = xgb.DMatrix(X_train, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)
#set colsampleby_tree and subsample set to 0.8 to prevent overfitting at this stage
#i set the eta as lower here for this run to see if i can avoid it overfitting
xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.8, 'gamma': 0, 'eta': 0.05, 'min_child_weight': 1, 'alpha': 0, 'lambda': 1, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.8, 'nthread': 4, 'seed': 42}
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
#83

#step 2
#Tune max_depth and min_child_weight
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 83),
    search_spaces = {
        'max_depth': (2, 15, 'uniform'),
        'min_child_weight': (2, 15, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 2,
    n_iter = 200,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
#Model #100
#Best pr-AUC: 0.6269
#Best params: OrderedDict([('max_depth', 14), ('min_child_weight', 5)])


#Tune gamma
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 83, max_depth = 14, min_child_weight = 5),
    search_spaces = {
        'gamma': (1e-12, 1.0, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 4,
    n_iter = 50,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
#Model #50
#Best pr-AUC: 0.6269
#Best params: OrderedDict([('gamma', 0.00010715710904743261)])

from sklearn.model_selection import cross_val_score
estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 83, max_depth = 14, min_child_weight = 5, gamma = 0)
np.mean(cross_val_score(estimator, X_train, y_train, cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15), scoring = 'average_precision'))
#0.6267262748515171


#tune subsample and colsample_bytree
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 83, max_depth = 14, min_child_weight = 5, gamma  = 0.00010715710904743261),
    search_spaces = {
        'colsample_bytree': (0.6, 1.0, 'uniform'),
        'subsample':  (0.6, 1.0, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 2,
    n_iter = 100,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
#Model #100
#Best pr-AUC: 0.6508
#Best params: OrderedDict([('colsample_bytree', 0.9977417734461554), ('subsample', 0.8452658867569602)])


#step 3

#tune gamma and alpha
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 83, max_depth = 14, min_child_weight = 5, gamma  = 0.00010715710904743261, colsample_bytree = 0.9977417734461554, subsample = 0.8452658867569602),
    search_spaces = {
        'reg_alpha': (1e-12, 5.0, 'log-uniform'),
        'reg_lambda':  (1e-12, 5.0, 'log-uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 2,
    n_iter = 100,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
#Model #100
#Best pr-AUC: 0.6401
#Best params: OrderedDict([('reg_alpha', 0.006284858393009234), ('reg_lambda', 8.971825843271272e-10)])

from sklearn.model_selection import cross_val_score
estimator = XGBClassifier(random_state=42, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 83, max_depth = 14, min_child_weight = 5, gamma  = 0.00010715710904743261, colsample_bytree = 0.9977417734461554, subsample = 0.8452658867569602, reg_alpha = 0.006284858393009234, reg_lambda = 8.971825843271272e-10)
np.mean(cross_val_score(estimator, X_train, y_train, cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15), scoring = 'average_precision'))
#0.6398337494310193
estimator = XGBClassifier(random_state=42, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 83, max_depth = 14, min_child_weight = 5, gamma  = 0.00010715710904743261, colsample_bytree = 0.9977417734461554, subsample = 0.8452658867569602, reg_alpha = 0.006284858393009234, reg_lambda = 0)
np.mean(cross_val_score(estimator, X_train, y_train, cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15), scoring = 'average_precision'))
#0.6398337494310193



#step 4 
#lower the training rate and find best
xgtrain = xgb.DMatrix(X_train, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)

results = []
lengths = []
etas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.9977417734461554, 'gamma': 0.00010715710904743261, 'eta': x, 'max_depth': 14, 'min_child_weight': 5,  'alpha': 0.006284858393009234, 'lambda': 0, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.8452658867569602, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.6571636, 0.6559288, 0.6552798, 0.6597074, 0.6603313999999999, 0.6478864, 0.62375]
#[289, 261, 53, 52, 51, 183, 51]


results = []
lengths = []
etas = [5e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3, 5e-3]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.9977417734461554, 'gamma': 0.00010715710904743261, 'eta': x, 'max_depth': 14, 'min_child_weight': 5,  'alpha': 0.006284858393009234, 'lambda': 0, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.8452658867569602, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=100, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.6593704, 0.6622498, 0.6617308000000001, 0.6603313999999999, 0.6540853999999999, 0.6597950000000001, 0.6364241999999999]
#[52, 53, 52, 51, 52, 52, 48]

results = []
lengths = []
etas = [6e-4, 7e-4, 8e-4]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.9977417734461554, 'gamma': 0.00010715710904743261, 'eta': x, 'max_depth': 14, 'min_child_weight': 5,  'alpha': 0.006284858393009234, 'lambda': 0, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.8452658867569602, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=100, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.6550575999999999, 0.6597500000000001, 0.6622498]
#[53, 52, 53]


############
#	Run the model
############
clf = XGBClassifier(random_state=42, scale_pos_weight = 1.148936170212766, learning_rate=8e-4, n_estimators = 53, max_depth = 14, min_child_weight = 5, gamma  = 0.00010715710904743261, colsample_bytree = 0.9977417734461554, subsample = 0.8452658867569602, reg_alpha = 0.006284858393009234, reg_lambda = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.5877862595419847
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.600     0.618     0.609        68
#           1      0.574     0.556     0.565        63
#    accuracy                          0.588       131
#   macro avg      0.587     0.587     0.587       131
#weighted avg      0.587     0.588     0.587       131


############
#	Plot the model evaluation
############
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\5_ML\\4_XGBoost')

probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.6783380018674136
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('XGBoost_ROC_bayesian_AllGenes.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.6789390137581166
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('XGBoost_prAUC_bayesian_AllGenes.png', dpi = 2000)

