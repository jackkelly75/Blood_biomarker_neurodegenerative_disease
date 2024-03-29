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

#########
# LASSO
#########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\4_FeatureSelection\\LASSO')

with open("X_train_LASSO", "rb") as input_file:
    X_train_LASSO = pickle.load(input_file)

with open("X_test_LASSO", "rb") as input_file:
    X_test_LASSO = pickle.load(input_file)


####################
# optimse model params using gridSearchCV
####################
#Step 1     -   High learning rate (0.1) and find best n_estimators (number of trees)
#Step 2     -   Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) for decided learning rate and number of trees
#Step 3     -   Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
#Step 4     -   Lower learning rate and retune n_estimators

##Step 1
xgtrain = xgb.DMatrix(X_train_LASSO, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)
#set colsampleby_tree and subsample set to 0.8 to prevent overfitting at this stage
#i set the eta as lower here for this run to see if i can avoid it overfitting
xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.8, 'gamma': 0, 'eta': 0.05, 'min_child_weight': 1, 'alpha': 0, 'lambda': 1, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.8, 'nthread': 4, 'seed': 42}
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
#361

#step 2
#Tune max_depth and min_child_weight
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 361),
    search_spaces = {
        'max_depth': (2, 15, 'uniform'),
        'min_child_weight': (2, 15, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 4,
    n_iter = 200,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_LASSO, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.7976
#Best params: OrderedDict([('max_depth', 2), ('min_child_weight', 2)])

#Tune gamma
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 361, max_depth = 2, min_child_weight = 2),
    search_spaces = {
        'gamma': (1e-12, 1.0, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 4,
    n_iter = 200,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_LASSO, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.8017
#Best params: OrderedDict([('gamma', 0.6607049016145172)])


#tune subsample and colsample_bytree
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 361, max_depth = 2, min_child_weight = 2, gamma = 0.6607049016145172),
    search_spaces = {
        'colsample_bytree': (0.2, 1.0, 'uniform'),
        'subsample':  (0.6, 1.0, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 4,
    n_iter = 200,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_LASSO, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.8084
#Best params: OrderedDict([('colsample_bytree', 0.7796245759127367), ('subsample', 0.709452824749872)])


#step 3
#tune gamma and alpha
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.7796245759127367, subsample = 0.709452824749872, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 361, max_depth = 2, min_child_weight = 2, gamma = 0.6607049016145172),
    search_spaces = {
        'reg_alpha': (1e-12, 5.0, 'uniform'),
        'reg_lambda':  (1e-12, 5.0, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 4,
    n_iter = 200,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_LASSO, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.8134
#Best params: OrderedDict([('reg_alpha', 0.0054108105270447805), ('reg_lambda', 0.06610763912281639)])


#step 4 
#lower the training rate and find best
xgtrain = xgb.DMatrix(X_train_LASSO, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)

results = []
lengths = []
etas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.7796245759127367, 'gamma': 0.6607049016145172, 'eta': x, 'max_depth': 2, 'min_child_weight': 2,  'alpha': 0.0054108105270447805, 'lambda': 0.06610763912281639, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.709452824749872, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.7442612000000001, 0.7447308000000001, 0.7462248, 0.7572319999999999, 0.8029374, 0.8122565999999999, 0.8191166000000001]
#[2755, 2736, 2658, 5000, 5000, 744, 91]

results = []
lengths = []
etas = [0.12, 0.11, 0.1, 0.09, 0.08]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.7796245759127367, 'gamma': 0.6607049016145172, 'eta': x, 'max_depth': 2, 'min_child_weight': 2,  'alpha': 0.0054108105270447805, 'lambda': 0.06610763912281639, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.709452824749872, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.8161346, 0.8100871999999999, 0.8191166000000001, 0.8114044, 0.8128572000000001]
#[94, 100, 91, 122, 121]
#best 0.1 - 91


############
#	Run the model 
############
clf = XGBClassifier(random_state=42, colsample_bytree = 0.7796245759127367, subsample = 0.709452824749872, scale_pos_weight = 1.148936170212766, learning_rate=0.1, n_estimators = 91, max_depth = 2, min_child_weight = 2, gamma = 0.6607049016145172, reg_alpha = 0.0054108105270447805, reg_lambda = 0.06610763912281639)
clf.fit(X_train_LASSO, y_train)
y_pred = clf.predict(X_test_LASSO)
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

probas_ = clf.fit(X_train_LASSO, y_train).predict_proba(X_test_LASSO)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.6286181139122317
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('XGBoost_ROC_bayesian_LASSO.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.6424232195291184
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('XGBoost_prAUC_bayesian_LASSO.png', dpi = 2000)
