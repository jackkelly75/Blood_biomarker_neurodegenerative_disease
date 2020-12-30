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
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

#####################
#	import the data
#####################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\3_scaled+sorted_data')
X_train = pd.read_csv('X_train.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('y_train.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('X_test.txt', sep="\t", header= 0, index_col=0)
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



##########
# Elastic Net
##########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\4_FeatureSelection\\ElasticNet')

with open("X_train_ElasticNet", "rb") as input_file:
    X_train_ElasticNet = pickle.load(input_file)

with open("X_test_ElasticNet", "rb") as input_file:
    X_test_ElasticNet = pickle.load(input_file)


############
#	Tune the model
############
bayes_cv_tuner = BayesSearchCV(
    estimator = LogisticRegression(random_state=2,  class_weight='balanced', penalty='l2', solver='liblinear'),
    search_spaces = {
        'C': (1e-3, 1e+4, 'log-uniform')
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

result = bayes_cv_tuner.fit(X_train_ElasticNet, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.8173
#Best params: OrderedDict([('C', 0.041907107241951104)])


############
#	Run the model
############
clf = LogisticRegression(C= 0.041907107241951104 , random_state=2,  class_weight='balanced', penalty='l2', solver='liblinear')
clf.fit(X_train_ElasticNet, y_train)
y_pred = clf.predict(X_test_ElasticNet)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.6030534351145038
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.608     0.662     0.634        68
#           1      0.596     0.540     0.567        63
#    accuracy                          0.603       131
#   macro avg      0.602     0.601     0.600       131
#weighted avg      0.603     0.603     0.602       131


############
#	Plot the model evaluation
############
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\5_ML\\1_Logistic_regression')

probas_ = clf.fit(X_train_ElasticNet, y_train).predict_proba(X_test_ElasticNet)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.6678338001867413
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('LogisticRegression_ROC_bayesian_ElasticNet.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.6756244555006613
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('LogisticRegression_prAUC_bayesian_ElasticNet.png', dpi = 2000)
