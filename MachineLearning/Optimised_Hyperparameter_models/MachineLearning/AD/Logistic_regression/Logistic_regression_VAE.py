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
    estimator = LogisticRegression(random_state=2,  class_weight='balanced', penalty='l2', solver='liblinear', max_iter = 5000),
    search_spaces = {
        'C': (1e-7, 1e+1, 'log-uniform')
    },    
     cv = StratifiedKFold(
        n_splits=5,
        shuffle = True,
        random_state=15
    ),
    n_jobs = 1,
    n_iter = 50,
    verbose = 0,
    refit = True,
    random_state = 5,
    scoring = 'average_precision'
)

result = bayes_cv_tuner.fit(X_train_VAE, y_train, callback=status_print)

#Model #50
#Best pr-AUC: 0.602
#Best params: OrderedDict([('C', 0.0016669407132495569)])


##########
# Run the model
##########
clf = LogisticRegression(C=0.0016669407132495569 , random_state=2,  class_weight='balanced', penalty='l2', solver='liblinear')
clf.fit(X_train_VAE, y_train)
y_pred = clf.predict(X_test_VAE)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy: 0.6477732793522267
# Model Precision: what percentage of positive tuples are labeled as such?
# Model Recall (sensitivity): what percentage of positive tuples are labelled as such?
#recall for 0 is specificity
#recall for 1 is sensitivity
print(metrics.classification_report(y_test, y_pred, digits = 3)) 
#              precision    recall  f1-score   support
#           0      0.574     0.635     0.603       104
#           1      0.712     0.657     0.684       143
#    accuracy                          0.648       247
#   macro avg      0.643     0.646     0.643       247
#weighted avg      0.654     0.648     0.650       247


##########
# Plot the results
##########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\AD_ML\\5_ML\\AD HC\\1_Logistic_regression')

probas_ = clf.fit(X_train_VAE, y_train).predict_proba(X_test_VAE)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
#0.6611081226465843
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('LogisticRegression_ROC_bayesian_VAE.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test, probas_[:, 1])
pr_auc = auc(recall, precision)
#0.6921079724025172
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('LogisticRegression_prAUC_bayesian_VAE.png', dpi = 2000)
