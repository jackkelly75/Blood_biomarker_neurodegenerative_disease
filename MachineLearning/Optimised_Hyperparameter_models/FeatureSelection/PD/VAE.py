#based on:
	#https://arxiv.org/pdf/1908.06278.pdf
	#https://github.com/zhangxiaoyu11/OmiVAE/blob/master/ExprOmiVAE.py

#####################
#	set seed to ensure at least some reproducibility with keras
#####################
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
#	import packages for feature selection
#####################
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import pickle
import numpy as np
from sklearn import metrics
from numpy import mean
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RepeatedKFold
from skopt import BayesSearchCV
from sklearn.utils import class_weight
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, precision_recall_curve, roc_curve, auc
import tensorflow as tf
from pandas import read_csv, DataFrame
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
import keras
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


#####################
#	import data
#####################
#   set wd and import file
os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\3_scaled+sorted_data')
X_train = pd.read_csv('X_train.txt', sep="\t", header= 0, index_col=0)
X_test = pd.read_csv('X_test.txt', sep="\t", header= 0, index_col=0)
y_train = pd.read_csv('y_train.txt', sep="\t", header= 0, index_col=0)
y_train = y_train.values.ravel()
y_test = pd.read_csv('y_test.txt', sep="\t", header= 0, index_col=0)
y_test = y_test.values.ravel()


######################
#   VAE
######################
# DEFINE THE ENCODER LAYERS
input_dim = Input(shape = (20183, ))
encoded1 = Dense(4096, activation = 'relu')(input_dim)
encoded1 = BatchNormalization()(encoded1)
encoded2 = Dense(1024, activation = 'relu')(encoded1)
encoded2 = BatchNormalization()(encoded2)
encoded2 = Dropout(0.2)(encoded2)
encoded3 = Dense(512, activation = 'relu')(encoded2)
encoded3 = BatchNormalization()(encoded3)
encoded3 = Dropout(0.2)(encoded3)

mu = Dense(128, name='latent_mu')(encoded3)
mu = BatchNormalization()(mu)
sigma = Dense(128, name='latent_sigma')(encoded3)
sigma = BatchNormalization()(sigma)

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sample_z(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    mu, sigma = args
    batch     = K.shape(mu)[0]
    dim       = K.int_shape(mu)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    eps       = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps

# use reparameterization trick to push the sampling out as input
z  = Lambda(sample_z, output_shape=(128, ), name='z')([mu, sigma])


# CLASSIFIER
classifier1 = Dense(128, activation = 'relu')(mu)
classifier1 = BatchNormalization()(classifier1)
classifier2 = Dense(64, activation = 'relu')(classifier1)
classifier2 = BatchNormalization()(classifier2)
pred_y = Dense(2, activation = 'softmax')(classifier2)

# DEFINE THE DECODER LAYERS
#decoded0 = keras.layers.BatchNormalization()(z)
decoded1 = Dense(512, activation = 'relu')(z)
decoded1 = BatchNormalization()(decoded1)
decoded2 = Dense(1024, activation = 'relu')(decoded1)
decoded2 = BatchNormalization()(decoded2)
decoded2 = Dropout(0.2)(decoded2)
decoded3 = Dense(4096, activation = 'relu')(decoded2)
decoded3 = BatchNormalization()(decoded3)
decoded3 = Dropout(0.2)(decoded3)
decoded4 = Dense(20183, activation = 'sigmoid')(decoded3)
decoded4 = BatchNormalization()(decoded4)

# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = keras.Model(input_dim, decoded4)

# CONFIGURE AND TRAIN THE AUTOENCODER
callbacks = [EarlyStopping(monitor='loss', patience=10),
             ModelCheckpoint(filepath='delete.h5', monitor='loss', save_best_only=True)]
opt = keras.optimizers.Adam(learning_rate=0.001)
autoencoder.compile(optimizer = opt, loss = 'binary_crossentropy')
history  = autoencoder.fit(X_train, X_train, epochs = 100, callbacks=callbacks, batch_size = 32, shuffle = True)

# THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
encoder = Model(input_dim, z, name='encoder')
encoder.summary()
encoded_input = Input(shape = (20183, ))
temp = encoder.predict(X_train)
encoded_out = encoder.predict(X_test)
encoded_out

os.chdir('C:\\Users\\Jack\\Desktop\\Deep_learning\\PD_ML_final\\4_FeatureSelection\\VAE')

with open('X_train_VAE', 'wb') as fp:
    pickle.dump(temp, fp)

with open('X_test_VAE', 'wb') as fp:
    pickle.dump(encoded_out, fp)


