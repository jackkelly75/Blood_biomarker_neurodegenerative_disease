
###set seed for keras work
# Seed value
# Apparently you may use different seed values at each stage
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


#####################
#	import packages
#####################
import os
import pandas as pd
import time
import pickle
import numpy as np
from numpy import mean
import pylab as pl
from sklearn.metrics import precision_score, recall_score, accuracy_score, precision_recall_curve, roc_curve, auc, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedKFold
import tensorflow as tf
tf.random.set_seed(17)
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import maxnorm
from keras import metrics
from matplotlib import pyplot as pl


#####################
#	import data
#####################
#import the data to be used (import feature selection datasets if using those)
X_train = pd.read_csv('X_train.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('X_test.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('y_train.txt', sep="\t", header= 0, index_col=0)
y_train = y_train.values.ravel()
y_test = pd.read_csv('y_test.txt', sep="\t", header= 0, index_col=0)
y_test = y_test.values.ravel()
X = X_train.values

from keras.utils import to_categorical
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)
count_classes = y_test1.shape[1]
print(count_classes)

feature_num = len(X_train.columns)

#################################
#	identifying best model
#	
#	below I have an example of a model that is being tested
#################################
VALIDATION_ACCURACY = []
VALIDATION_LOSS = []
VALIDATION_ROC = []
VALIDATION_PR = []
skf = StratifiedKFold(n_splits = 5, random_state = 15, shuffle = True) 
fold_var = 1

for train_index, val_index in skf.split(X,y_train):
	training_data = X[train_index]
	validation_data = X[val_index]
	training_y = y_train[train_index]
	validation_y = y_train[val_index]
	training_y = to_categorical(training_y)
	validation_y = to_categorical(validation_y)

	model = Sequential()
	model.add(Dense(4096, activation = 'relu', input_dim=feature_num))
	model.add(BatchNormalization())
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(BatchNormalization())
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	# Compile the model
	opt = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC(name='AUC', curve ='ROC'),metrics.AUC(name='PR', curve ='PR')])
	callbacks = [EarlyStopping(monitor='PR', patience=50), ModelCheckpoint(filepath='best_model.h5', monitor='PR', save_best_only=True)]
	print('------------------------------------------------------------------------')
	print(f'Training for fold {fold_var} ...')
	history = model.fit(training_data, training_y, epochs=100, callbacks = callbacks, batch_size = 32, validation_data = (validation_data, validation_y))

	results = model.evaluate(validation_data, validation_y, verbose=0)
	results = dict(zip(model.metrics_names,results))
	VALIDATION_ACCURACY.append(results['accuracy'])
	VALIDATION_LOSS.append(results['loss'])
	VALIDATION_ROC.append(results['AUC'])
	VALIDATION_PR.append(results['PR'])
	fold_var += 1

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(VALIDATION_ACCURACY)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {VALIDATION_LOSS[i]} - Accuracy: {VALIDATION_ACCURACY[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(VALIDATION_ACCURACY)} (+- {np.std(VALIDATION_ACCURACY)})')
print(f'> Loss: {np.mean(VALIDATION_LOSS)}')
print('------------------------------------------------------------------------')

model.summary()





#################
#Best model
#################
#the best model found above is run here

model = Sequential()
model.add(Dense(4096, activation = 'relu', input_dim=feature_num))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC(name='AUC', curve ='ROC'),metrics.AUC(name='PR', curve ='PR')])
callbacks = [EarlyStopping(monitor='PR', patience=3), ModelCheckpoint(filepath='best_model.h5', monitor='PR', save_best_only=True)]
history = model.fit(X, y_train1, epochs=100, callbacks = callbacks, batch_size = 32)



# Plot history
pl.plot(history.history['accuracy'], color="orange", label = "Accuracy")
pl.plot(history.history['loss'], color="slateblue", label = "Loss")
pl.plot(history.history['PR'], color="crimson", label = "pr_AUC")
pl.title('Validation accuracy history')
pl.ylabel('Score')
pl.xlabel('No. epoch')
positions = [9, 19, 29, 39, 49]
labels = [10, 20, 30, 40, 50]
pl.xticks(positions, labels)
pl.legend()
pl.savefig('Validation_history.png', dpi = 2000)
pl.clf()


y_pred = model.predict(X_test.values)
y_pos = np.array([row[1] for row in y_pred])
y_pred = np.argmax(y_pred, axis=1)
print("Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred)) 


#plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pos)
roc_auc = auc(fpr, tpr)
print(roc_auc)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.savefig('VAE_ROCAUC.png', dpi = 2000)

#plot pr curve
precision, recall, _  = precision_recall_curve(y_test, y_pos)
pr_auc = auc(recall, precision)
print(pr_auc)
pl.clf()
pl.plot(recall, precision, label='Precision recall curve (area = %0.2f)' % pr_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.title('Precision recall curve')
pl.legend(loc="lower right")
pl.savefig('VAE_prAUC.png', dpi = 2000)


model.save('my_model.h5') 


##################
#can reload the model later using
##################
new_model = tf.keras.models.load_model('my_model.h5')
