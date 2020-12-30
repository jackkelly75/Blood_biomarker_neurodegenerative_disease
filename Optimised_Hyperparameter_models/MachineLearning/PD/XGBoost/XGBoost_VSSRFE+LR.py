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
# VSSRFE + LinearRegression
#########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\4_FeatureSelection\\VSSRFE+LinearRegression')

with open("X_train_VSSRFE_LR", "rb") as input_file:
    X_train_VSSRFE_LR = pickle.load(input_file)

with open("X_test_VSSRFE_LR", "rb") as input_file:
    X_test_VSSRFE_LR = pickle.load(input_file)

####################
# optimse model params using gridSearchCV
####################
#Step 1     -   High learning rate (0.1) and find best n_estimators (number of trees)
#Step 2     -   Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) for decided learning rate and number of trees
#Step 3     -   Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
#Step 4     -   Lower learning rate and retune n_estimators

##Step 1
xgtrain = xgb.DMatrix(X_train_VSSRFE_LR, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)
#set colsampleby_tree and subsample set to 0.8 to prevent overfitting at this stage
#i set the eta as lower here for this run to see if i can avoid it overfitting
xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.8, 'gamma': 0, 'eta': 0.05, 'min_child_weight': 1, 'alpha': 0, 'lambda': 1, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.8, 'nthread': 4, 'seed': 42}
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
#15

#step 2
#Tune max_depth and min_child_weight
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 15),
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

result = bayes_cv_tuner.fit(X_train_VSSRFE_LR, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.6477
#Best params: OrderedDict([('max_depth', 2), ('min_child_weight', 4)])


#Tune gamma
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 15, max_depth = 2, min_child_weight = 4),
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

result = bayes_cv_tuner.fit(X_train_VSSRFE_LR, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.6477
#Best params: OrderedDict([('gamma', 0.08819901280121412)])



#tune subsample and colsample_bytree
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 15, max_depth = 2, min_child_weight = 4, gamma = 0.08819901280121412),
    search_spaces = {
        'colsample_bytree': (0.6, 1.0, 'uniform'),
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

result = bayes_cv_tuner.fit(X_train_VSSRFE_LR, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.6705
#Best params: OrderedDict([('colsample_bytree', 0.6007399925272013), ('subsample', 0.9669266113540623)])

#step 3

#tune gamma and alpha
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, scale_pos_weight = 1.148936170212766, learning_rate=0.05, n_estimators = 15, max_depth = 2, min_child_weight = 4, gamma = 0.08819901280121412, colsample_bytree = 0.6007399925272013, subsample = 0.9669266113540623),
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

result = bayes_cv_tuner.fit(X_train_VSSRFE_LR, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.6719
#Best params: OrderedDict([('reg_alpha', 0.0028469155578777577), ('reg_lambda', 0.859930396483553)])


#step 4 
#lower the training rate and find best
xgtrain = xgb.DMatrix(X_train_VSSRFE_LR, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)

results = []
lengths = []
etas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.6007399925272013, 'gamma': 0.08819901280121412, 'eta': x, 'max_depth': 2, 'min_child_weight': 4,  'alpha': 0.0028469155578777577, 'lambda': 0.859930396483553, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.9669266113540623, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.6651583999999999, 0.6654412, 0.6665627999999999, 0.6689396000000001, 0.6755504, 0.6735214, 0.6778088]
#[2018, 2930, 2926, 5000, 911, 66, 24]

results = []
lengths = []
etas = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.6007399925272013, 'gamma': 0.08819901280121412, 'eta': x, 'max_depth': 2, 'min_child_weight': 4,  'alpha': 0.0028469155578777577, 'lambda': 0.859930396483553, 'scale_pos_weight': 1.148936170212766, 'subsample': 0.9669266113540623, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.6744158, 0.6731316, 0.6716355999999999, 0.6706828, 0.6756367999999999, 0.6778088, 0.67754, 0.6711503999999999]
#[8, 25, 25, 24, 19, 24, 9, 6]
#best 0.1, 24


############
#	Run the model
############
clf = XGBClassifier(random_state=42, scale_pos_weight = 1.148936170212766, learning_rate=0.1, n_estimators = 24, max_depth = 2, min_child_weight = 4, gamma = 0.08819901280121412, colsample_bytree = 0.6007399925272013, subsample = 0.9669266113540623, reg_alpha = 0.0028469155578777577, reg_lambda = 0.859930396483553)
clf.fit(X_train_VSSRFE_LR, y_train)
y_pred = clf.predict(X_test_VSSRFE_LR)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.6793893129770993
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.671     0.750     0.708        68
#           1      0.691     0.603     0.644        63
#    accuracy                          0.679       131
#   macro avg      0.681     0.677     0.676       131
#weighted avg      0.681     0.679     0.677       131


############
#	Plot the model evaluation
############
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\5_ML\\4_XGBoost')

probas_ = clf.fit(X_train_VSSRFE_LR, y_train).predict_proba(X_test_VSSRFE_LR)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.6814892623716152
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('XGBoost_ROC_bayesian_VSSRFE_LR.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.6746022342473319
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('XGBoost_prAUC_bayesian_VSSRFE_LR.png', dpi = 2000)

