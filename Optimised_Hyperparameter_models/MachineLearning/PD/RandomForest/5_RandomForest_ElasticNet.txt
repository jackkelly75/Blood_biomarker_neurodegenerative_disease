#####################
#	import packages
#####################
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc, precision_recall_curve
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as pl

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



#########
# ElasticNet
#########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\4_FeatureSelection\\ElasticNet')

with open("X_train_ElasticNet", "rb") as input_file:
    X_train_ElasticNet = pickle.load(input_file)

with open("X_test_ElasticNet", "rb") as input_file:
    X_test_ElasticNet = pickle.load(input_file)


############
#	Tune the model
############
bayes_cv_tuner = BayesSearchCV(
    estimator = RandomForestClassifier(random_state=10,  class_weight='balanced'),
    search_spaces = {
        'max_depth': (2, 15, 'uniform'),
        'min_samples_leaf': (1, 15, 'uniform')
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

result = bayes_cv_tuner.fit(X_train_ElasticNet, y_train, callback=status_print)
#Model #200
#Best pr-AUC: 0.8023
#Best params: OrderedDict([('max_depth', 11), ('min_samples_leaf', 1)])

bayes_cv_tuner = BayesSearchCV(
    estimator = RandomForestClassifier(random_state=10,  class_weight='balanced', max_depth = 11, min_samples_leaf = 1),
    search_spaces = {
        'n_estimators': (10, 5000, 'uniform')
    },    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 3,
    n_iter = 100,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_ElasticNet, y_train, callback=status_print)
#Model #100
#Best pr-AUC: 0.7924
#Best params: OrderedDict([('n_estimators', 837)])

bayes_cv_tuner = BayesSearchCV(
    estimator = RandomForestClassifier(random_state=10,  class_weight='balanced', max_depth = 11, min_samples_leaf = 1, n_estimators = 837),
    search_spaces = {
        'min_samples_split': (2, 150, 'uniform'),
        'max_features': (1, 24, 'uniform')
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

result = bayes_cv_tuner.fit(X_train_ElasticNet, y_train, callback=status_print)
#Model #50
#Best pr-AUC: 0.806
#Best params: OrderedDict([('max_features', 1), ('min_samples_split', 18)])


############
#	Run the model
############
clf = RandomForestClassifier(random_state=10,  class_weight='balanced', max_depth = 11, min_samples_leaf = 1, n_estimators = 837, max_features = 1, min_samples_split = 18)
clf.fit(X_train_ElasticNet, y_train)
y_pred = clf.predict(X_test_ElasticNet)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.648854961832061
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.641     0.735     0.685        68
#           1      0.660     0.556     0.603        63
#    accuracy                          0.649       131
#   macro avg      0.651     0.645     0.644       131
#weighted avg      0.650     0.649     0.646       131


############
#	Plot the model evaluation
############
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\5_ML\\5_RandomForest')

probas_ = clf.fit(X_train_ElasticNet, y_train).predict_proba(X_test_ElasticNet)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.6622315592903829
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('RandomForest_ROC_bayesian_ElasticNet.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.6629869185480943
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('RandomForest_prAUC_bayesian_ElasticNet.png', dpi = 2000)
