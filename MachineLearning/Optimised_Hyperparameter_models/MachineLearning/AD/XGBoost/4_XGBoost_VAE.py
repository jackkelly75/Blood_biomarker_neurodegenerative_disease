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
# VAE
#########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\4_FeatureSelection\\AD HC\\VAE')

with open("X_train_VAE", "rb") as input_file:
    X_train_VAE = pickle.load(input_file)

with open("X_test_VAE", "rb") as input_file:
    X_test_VAE = pickle.load(input_file)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_VAE = scaler.fit_transform(X_train_VAE)
X_test_VAE = scaler.fit_transform(X_test_VAE)


####################
# optimse model params using gridSearchCV
####################
#Step 1     -   High learning rate (0.1) and find best n_estimators (number of trees)
#Step 2     -   Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) for decided learning rate and number of trees
#Step 3     -   Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
#Step 4     -   Lower learning rate and retune n_estimators


##Step 1
xgtrain = xgb.DMatrix(X_train_VAE, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)
#set colsampleby_tree and subsample set to 0.8 to prevent overfitting at this stage
#i set the eta as lower here for this run to see if i can avoid it overfitting
xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.8, 'gamma': 0, 'eta': 0.05, 'min_child_weight': 1, 'alpha': 0, 'lambda': 1, 'scale_pos_weight': 0.9562043795620438, 'subsample': 0.8, 'nthread': 4, 'seed': 42}
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=2000, verbose_eval = True, seed = 5)
#31

#step 2
#Tune max_depth and min_child_weight
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 0.9562043795620438, learning_rate=0.05, n_estimators = 31),
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

result = bayes_cv_tuner.fit(X_train_VAE, y_train, callback=status_print)

#Model #20
#Best pr-AUC: 0.6508
#Best params: OrderedDict([('max_depth', 7), ('min_child_weight', 4)])


#Tune gamma
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 0.9562043795620438, learning_rate=0.05, n_estimators = 31, max_depth = 7, min_child_weight = 4),
    search_spaces = {
        'gamma': (1e-12, 10.0, 'log-uniform')
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

result = bayes_cv_tuner.fit(X_train_VAE, y_train, callback=status_print)

#Model #50
#Best pr-AUC: 0.6487
#Best params: OrderedDict([('gamma', 0.5840810014249553)])


#tune subsample and colsample_bytree
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.8, subsample = 0.8, scale_pos_weight = 0.9562043795620438, learning_rate=0.05, n_estimators = 31, max_depth = 7, min_child_weight = 4, gamma = 0.5840810014249553),
    search_spaces = {
        'colsample_bytree': (0.2, 1.0, 'uniform'),
        'subsample':  (0.2, 1.0, 'uniform')
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

result = bayes_cv_tuner.fit(X_train_VAE, y_train, callback=status_print)

#Model #50
#Best pr-AUC: 0.644
#Best params: OrderedDict([('colsample_bytree', 0.5195812889570743), ('subsample', 0.6816009688535446)])



#step 3
#tune gamma and alpha
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(random_state=42, colsample_bytree = 0.5195812889570743, subsample = 0.6816009688535446, scale_pos_weight = 0.9562043795620438, learning_rate=0.05, n_estimators = 31, max_depth = 7, min_child_weight = 4, gamma = 0.5840810014249553),
    search_spaces = {
        'reg_alpha': (1e-12, 10.0, 'log-uniform'),
        'reg_lambda':  (1e-12, 10.0, 'log-uniform')
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

result = bayes_cv_tuner.fit(X_train_VAE, y_train, callback=status_print)

#Model #30
#Best pr-AUC: 0.6469
#Best params: OrderedDict([('reg_alpha', 1.906111100841027e-12), ('reg_lambda', 9.991851172872426)])


#step 4 
#lower the training rate and find best
xgtrain = xgb.DMatrix(X_train_VAE, y_train)
cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15)

results = []
lengths = []
etas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.5195812889570743, 'gamma': 0.5840810014249553, 'eta': x, 'max_depth': 7, 'min_child_weight': 4,  'alpha': 0, 'lambda': 9.991851172872426, 'scale_pos_weight': 0.9562043795620438, 'subsample': 0.6816009688535446, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=50, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.6304858000000001, 0.6415749999999999, 0.6589246000000001, 0.659, 0.6563384, 0.6637992, 0.6403116]
#[232, 35, 13, 13, 13, 21, 4]


results = []
lengths = []
etas = [5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.5195812889570743, 'gamma': 0.5840810014249553, 'eta': x, 'max_depth': 7, 'min_child_weight': 4,  'alpha': 0, 'lambda': 9.991851172872426, 'scale_pos_weight': 0.9562043795620438, 'subsample': 0.6816009688535446, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=50, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.6675232, 0.665029, 0.6651576, 0.663593, 0.664426, 0.6637992, 0.6585496, 0.6610798, 0.6467094, 0.6445418]
#[21, 21, 21, 21, 21, 21, 30, 31, 32, 35]


results = []
lengths = []
etas = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
for x in etas:
    xgb_param = {'objective': 'binary:logistic', 'colsample_bytree': 0.5195812889570743, 'gamma': 0.5840810014249553, 'eta': x, 'max_depth': 7, 'min_child_weight': 4,  'alpha': 0, 'lambda': 9.991851172872426, 'scale_pos_weight': 0.9562043795620438, 'subsample': 0.6816009688535446, 'seed': 42}
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, folds=cv, metrics='map', early_stopping_rounds=50, verbose_eval = True, seed = 5)
    results.append(cvresult['test-map-mean'].max())
    lengths.append(len(cvresult))
#[0.6563384, 0.6563384, 0.6563384, 0.6680322000000001, 0.6675232]
#[13, 13, 13, 21, 21]


#####################
#   Run the model
#####################
clf = XGBClassifier(random_state=42, colsample_bytree = 0.5195812889570743, subsample = 0.6816009688535446, scale_pos_weight = 0.9562043795620438, learning_rate=4e-3, n_estimators = 21, max_depth = 7, min_child_weight = 4, gamma = 0.5840810014249553, reg_alpha = 0, reg_lambda= 9.991851172872426)
clf.fit(X_train_VAE, y_train)
y_pred = clf.predict(X_test_VAE)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.6275303643724697
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.603     0.337     0.432       104
#           1      0.635     0.839     0.723       143
#    accuracy                          0.628       247
#   macro avg      0.619     0.588     0.577       247
#weighted avg      0.622     0.628     0.600       247


#####################
#   Plot the model evaluation
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\5_ML\\AD HC\\3_XGBoost')

probas_ = clf.fit(X_train_VAE, y_train).predict_proba(X_test_VAE)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.6600658956428186
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('XGBoost_ROC_bayesian_VAE.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.7087028863251501
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('XGBoost_prAUC_bayesian_VAE.png', dpi = 2000)

