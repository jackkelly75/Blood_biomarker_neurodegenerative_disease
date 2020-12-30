#based ont his paper
# https://bmcmedgenomics.biomedcentral.com/articles/10.1186/s12920-020-0677-2
#https://github.com/chenlabgccri/CancerTypePrediction/blob/master/5cv_33class/5cv_1D_CNN_33class.py

###set seed for keras work
# Seed value
seed_value= 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

###############
#	import packages
###############
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import pickle
import numpy as np
from sklearn import metrics
from numpy import mean, array, argmax
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RepeatedKFold
from skopt import BayesSearchCV
from sklearn.utils import class_weight
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, precision_recall_curve, roc_curve, auc, average_precision_score
import tensorflow as tf
from pandas import read_csv, DataFrame
from numpy.random import seed
from sklearn.preprocessing import minmax_scale, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.models import Model, Sequential
import keras
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import collections
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as pl

#########
#	Import data
#########
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\3_scaled+sorted_data')
X_train = pd.read_csv('X_train.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('X_test.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('y_train.txt', sep="\t", header= 0, index_col=0)
y_train = y_train.values.ravel()
y_test = pd.read_csv('y_test.txt', sep="\t", header= 0, index_col=0)
y_test = y_test.values.ravel()
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)
count_classes = y_test1.shape[1]
print(count_classes)

#labels
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y_train.reshape(len(y_train), 1)
y_train = onehot_encoder.fit_transform(integer_encoded)
integer_encoded = y_test.reshape(len(y_test), 1)
y_test = onehot_encoder.fit_transform(integer_encoded)


#data
x_train = X_train.values
#add 0s to the end of the train to make it divsible by 100
x_train = np.concatenate((x_train,np.zeros((len(x_train),17))),axis=1)
x_train = np.reshape(x_train, (-1, 100, 202))

x_test = X_test.values
x_test = np.concatenate((x_test,np.zeros((len(x_test),17))),axis=1)
x_test = np.reshape(x_test, (-1, 100, 202))


##########
#	Set parameters for CNN
##########
img_rows, img_cols = len(x_train[0]), len(x_train[0][0])
num_classes = len(y_train[0])
batch_size = 32
epochs = 20
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

input_Xs = x_train
y_s = y_train
input_Xs = input_Xs.reshape(input_Xs.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
input_Xs = input_Xs.astype('float32')

input_Xs1 = x_test
input_Xs1 = input_Xs1.reshape(input_Xs1.shape[0], img_rows, img_cols, 1)
input_shape1 = (img_rows, img_cols, 1)
input_Xs1 = input_Xs1.astype('float32')


############
#	Define the model to be optimsed
############
def make_model(dense_layer_sizes, filters, kernel_size):
    model = Sequential()
    ## *********** First layer Conv
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(1, 2))
    model.output_shape
    ## ********* Classification layer
    model.add(Flatten())
    model.add(Dense(dense_layer_sizes, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.output_shape
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['categorical_accuracy'])
    model.summary()
    return model


####################
#	Optimise the model using grid search
####################
dense_size_candidates = [2, 8, 32, 64, 128, 512]
my_classifier = KerasClassifier(make_model, batch_size=32)
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # epochs is avail for tuning even when not
                                     # an argument to model building function
                                     'epochs': [25],
                                     'filters': [2, 4, 8, 16, 32, 64],
                                     'kernel_size': [(1,32), (1,100), (1,202), (1,300)]},
                         scoring='average_precision',
                         n_jobs=2, verbose = 2,
                         cv = KFold( n_splits=5, shuffle = True, random_state=15))
validator.fit(input_Xs, y_train)
print('The parameters of the best model are: ')
print(validator.best_params_)
#{'dense_layer_sizes': 512, 'epochs': 25, 'filters': 16, 'kernel_size': (1, 202)}
#0.6291228641645512
# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names

metric_values = best_model.evaluate(input_Xs1, y_test, verbose=0)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)


####################
#	Best Model
####################

model = Sequential()
## *********** First layer Conv
model.add(Conv2D(16, kernel_size=(1, 202), strides=(1, 1), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(1, 2))
## ********* Classification layer
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['categorical_accuracy'])
callbacks = [EarlyStopping(monitor='categorical_accuracy', patience=3, verbose=0)]
model.summary()

history = model.fit(input_Xs, y_train, batch_size=batch_size, epochs=100, verbose=0, callbacks=callbacks)

scores = model.evaluate(input_Xs1, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#categorical_accuracy: 69.47%


##################
# Save the model
##################
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\5_ML\\Convolutional_neural_network')

model.save('sgd_model.h5') 


##################
#reload the model
##################
new_model = tf.keras.models.load_model('sgd_model.h5')


##################
# Get model evaluation
##################
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_pred_keras = new_model.predict(input_Xs1)
print("Accuracy:",metrics.accuracy_score(y_test[:,1], np.round(y_pred_keras[:,1]))) 
#0.6946564885496184
print(metrics.classification_report(y_test[:,1], np.round(y_pred_keras[:,1]), digits = 3)) 
#              precision    recall  f1-score   support
#         0.0      0.700     0.721     0.710        68
#         1.0      0.689     0.667     0.677        63
#    accuracy                          0.695       131
#   macro avg      0.694     0.694     0.694       131
#weighted avg      0.694     0.695     0.694       131


##################
#	Plot model evaluation
##################
fpr, tpr, thresholds = roc_curve(y_test[:,1], y_pred_keras[:,1])
roc_auc = auc(fpr, tpr)
#0.7152194211017742
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('Convolutional_NN_ROC_bayesian_AllGenes.png', dpi = 2000)

precision, recall, _  = precision_recall_curve(y_test[:,1], y_pred_keras[:,1])
pr_auc = auc(recall, precision)
#0.709776074972908
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('Convolutional_NN_prAUC_bayesian_AllGenes.png', dpi = 2000)

