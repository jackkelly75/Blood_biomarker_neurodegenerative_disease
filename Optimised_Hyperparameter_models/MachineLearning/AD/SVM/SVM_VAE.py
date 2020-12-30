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
from sklearn.svm import SVC
from matplotlib import pyplot

#####################
#	import the data
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\3_scaled+sorted_data\\AD HC')
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
    print('Model #{}\nBest pr-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))



##########
# VAE
##########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\4_FeatureSelection\\AD HC\\VAE')

with open("X_train_VAE", "rb") as input_file:
    X_train_VAE = pickle.load(input_file)

with open("X_test_VAE", "rb") as input_file:
    X_test_VAE = pickle.load(input_file)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_VAE = scaler.fit_transform(X_train_VAE)
X_test_VAE = scaler.fit_transform(X_test_VAE)


bayes_cv_tuner = BayesSearchCV(
    estimator = SVC(random_state=142, kernel = 'rbf', class_weight = 'balanced'),
    search_spaces = {
        'C': (1e-4, 1e+2, 'log-uniform'),
        'gamma': (1e-4, 1e+2, 'log-uniform')
    },    
     cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 1,
    n_iter = 100,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_VAE, y_train, callback=status_print)
#Model #100
#Best pr-AUC: 0.6579
#Best params: OrderedDict([('C', 3.4028684317983213), ('gamma', 0.021770708275534352)])


#####################
#   Run the model
#####################
clf = SVC(kernel='rbf', probability=True, C=  3.4028684317983213, gamma = 0.021770708275534352,  random_state=142, class_weight = 'balanced')
clf.fit(X_train_VAE, y_train)
y_pred = clf.predict(X_test_VAE)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.5789473684210527
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.500     0.692     0.581       104
#           1      0.689     0.497     0.577       143
#    accuracy                          0.579       247
#   macro avg      0.595     0.594     0.579       247
#weighted avg      0.610     0.579     0.579       247


#####################
#   Plot the model evaluation
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\5_ML\\AD HC\\2_SVM')

probas_ = clf.fit(X_train_VAE, y_train).predict_proba(X_test_VAE)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
# 0.6147794513179129
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('SVM_ROC_bayesian_VAE.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.660582238705143
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('SVM_prAUC_bayesian_VAE.png', dpi = 2000)
