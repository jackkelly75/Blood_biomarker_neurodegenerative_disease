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
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\3_scaled+sorted_knowledge')
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
    n_iter = 100,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
#Model #100
#Best pr-AUC: 0.5976
#Best params: OrderedDict([('max_depth', 7), ('min_samples_leaf', 9)])


bayes_cv_tuner = BayesSearchCV(
    estimator = RandomForestClassifier(random_state=10,  class_weight='balanced', max_depth = 7, min_samples_leaf = 9),
    search_spaces = {
        'n_estimators': (10, 5000, 'uniform')
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
#Best pr-AUC: 0.5937
#Best params: OrderedDict([('n_estimators', 2477)])


bayes_cv_tuner = BayesSearchCV(
    estimator = RandomForestClassifier(random_state=10,  class_weight='balanced', max_depth = 7, min_samples_leaf = 9, n_estimators = 2477),
    search_spaces = {
        'min_samples_split': (2, 150, 'uniform')
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

result = bayes_cv_tuner.fit(X_train, y_train, callback=status_print)
#Model #50
#Best pr-AUC: 0.5937
#Best params: OrderedDict([('min_samples_split', 15)])


from sklearn.model_selection import cross_val_score
estimator = RandomForestClassifier(random_state=10,  class_weight='balanced', max_depth = 7, min_samples_leaf = 9, n_estimators = 2477, max_features = 'log2')
np.mean(cross_val_score(estimator, X_train, y_train, cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15), scoring = 'average_precision'))
#0.5570496826698523
estimator = RandomForestClassifier(random_state=10,  class_weight='balanced', max_depth = 7, min_samples_leaf = 9, n_estimators = 2477, max_features = 'sqrt')
np.mean(cross_val_score(estimator, X_train, y_train, cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=15), scoring = 'average_precision'))
#0.5936937797547095


############
#	Run the model
############
clf = RandomForestClassifier(random_state=10,  class_weight='balanced', max_depth = 7, min_samples_leaf = 9, n_estimators = 2477, max_features = 'sqrt')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.6717557251908397
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.637     0.853     0.730        68
#           1      0.750     0.476     0.583        63
#    accuracy                          0.672       131
#   macro avg      0.694     0.665     0.656       131
#weighted avg      0.692     0.672     0.659       131


############
#	Plot the model evaluation
############
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\5_ML\\5_RandomForest')

probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.7163865546218489
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('RandomForest_ROC_bayesian_KnowledgeGenes.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.7373789150555214
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('RandomForest_prAUC_bayesian_KnowledgeGenes.png', dpi = 2000)

