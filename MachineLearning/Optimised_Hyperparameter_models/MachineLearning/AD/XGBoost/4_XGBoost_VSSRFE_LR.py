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
#   import the data
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\3_scaled+sorted_data\\AD HC')
X_train = pd.read_csv('GSE63061.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('gse63061_pData.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('GSE63060.txt', sep="\t", header= 0, index_col=0)
y_test = pd.read_csv('gse63060_pData.txt', sep="\t", header= 0, index_col=0)

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
    print('Model #{}\nBest pr-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

(unique, counts) = numpy.unique(y_train, return_counts=True)
frequencies = numpy.asarray((unique, counts)).T
#[  0, 131]
#[  1, 137]
131/137
#0.9562043795620438




#########
# VSSRFE+LR
#########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\4_FeatureSelection\\AD HC\\VSSRFE+LinearRegression')

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
xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.8, 'gamma': 0, 'eta': 0.05, 'min_child_weight': 1, 'alpha': 0, 'lambda': 1, 'scale_pos_weight': 0.9562043795620438, 'subsample': 0.8, 'nthread': 4, 'seed': 42}
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
#2271

#step 2
#Tune max_depth and min_child_weight
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 0.9562043795620438, learning_rate=0.05, n_estimators = 2271),
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
    n_iter = 20,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_VSSRFE_LR, y_train, callback=status_print)

#Model #20
#Best pr-AUC: 0.9155
#Best params: OrderedDict([('max_depth', 13), ('min_child_weight', 2)])


#Tune gamma
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 0.9562043795620438, learning_rate=0.05, n_estimators = 2271, max_depth = 13, min_child_weight = 2),
    search_spaces = {
        'gamma': (1e-12, 1.0, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 2,
    n_iter = 50,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_VSSRFE_LR, y_train, callback=status_print)

#Model #50
#Best pr-AUC: 0.9196
#Best params: OrderedDict([('gamma', 0.16713401469455924)])

#tune subsample and colsample_bytree
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 0.9562043795620438, learning_rate=0.05, n_estimators = 2271, max_depth = 13, min_child_weight = 2, gamma = 0.16713401469455924),
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

result = bayes_cv_tuner.fit(X_train_VSSRFE_LR, y_train, callback=status_print)

#Model #100
#Best pr-AUC: 0.9353
#Best params: OrderedDict([('colsample_bytree', 0.6067127231029593), ('subsample', 0.600327569498154)])


#step 3
#tune gamma and alpha
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.6067127231029593, subsample = 0.600327569498154, scale_pos_weight = 0.9562043795620438, learning_rate=0.05, n_estimators = 2271, max_depth = 13, min_child_weight = 2, gamma = 0.16713401469455924),
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
    n_iter = 30,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_VSSRFE_LR, y_train, callback=status_print)

#Model #30
#Best pr-AUC: 0.9342
#Best params: OrderedDict([('reg_alpha', 0.0043058784736429035), ('reg_lambda', 1e-12)])


#step 4 
#lower the training rate and find best
xgtrain = xgb.DMatrix(X_train_VSSRFE_LR, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)

results = []
lengths = []
etas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.6067127231029593, 'gamma': 0.16713401469455924, 'eta': x, 'max_depth': 13, 'min_child_weight': 2,  'alpha': 0.0043058784736429035, 'lambda': 0, 'scale_pos_weight': 0.9562043795620438, 'subsample': 0.600327569498154, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=50, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.8864428, 0.892294, 0.8930118, 0.8937289999999999, 0.8924434, 0.9011517999999998, 0.9216214]
#[64, 50, 50, 46, 51, 42, 160]

results = []
lengths = []
etas = [6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1 ]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.6067127231029593, 'gamma': 0.16713401469455924, 'eta': x, 'max_depth': 13, 'min_child_weight': 2,  'alpha': 0.0043058784736429035, 'lambda': 0, 'scale_pos_weight': 0.9562043795620438, 'subsample': 0.600327569498154, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=50, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.9221104, 0.9169698000000001, 0.9251294000000001, 0.9110548000000002, 0.9216214, 0.927168, 0.9023783999999999, 0.867083, 0.8383136,0.8752409999999999, 0.8459968]
#[256, 49, 271, 93, 160, 202, 139, 38, 39, 31, 28]


#####################
#   Run the model
#####################
clf = XGBClassifier(random_state=42, colsample_bytree = 0.6067127231029593, subsample = 0.600327569498154, scale_pos_weight = 0.9562043795620438, learning_rate=2e-1, n_estimators = 202, max_depth = 13, min_child_weight = 2, gamma = 0.16713401469455924, reg_alpha = 0.0043058784736429035, reg_lambda = 0)
clf.fit(X_train_VSSRFE_LR, y_train)
y_pred = clf.predict(X_test_VSSRFE_LR)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.7935222672064778
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.779     0.712     0.744       104
#           1      0.803     0.853     0.827       143
#    accuracy                          0.794       247
#   macro avg      0.791     0.782     0.785       247
#weighted avg      0.793     0.794     0.792       247


#####################
#   Plot the model evaluation
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\5_ML\\AD HC\\3_XGBoost')

probas_ = clf.fit(X_train_VSSRFE_LR, y_train).predict_proba(X_test_VSSRFE_LR)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.8468262506724045
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('XGBoost_ROC_bayesian_VSSRFE+LR.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.8825797820888799
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('XGBoost_prAUC_bayesian_VSSRFE+LR.png', dpi = 2000)

