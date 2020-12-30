###Models
#1 - Normal
#2 - Batch normalisation
#3 - Dropout
#4 - Dropout + MaxNorm

#########
#	1	#
#########
##########
# Normal
##########
	model = Sequential()
	model.add(Dense(4096, activation = 'relu', input_dim=19147))
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(128))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	# Compile the model
	opt = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC(name='AUC', curve ='ROC'),metrics.AUC(name='PR', curve ='PR')])
	callbacks = [EarlyStopping(monitor='PR', patience=3), ModelCheckpoint(filepath='best_model.h5', monitor='PR', save_best_only=True)]
	print('------------------------------------------------------------------------')
	print(f'Training for fold {fold_var} ...')
	history = model.fit(training_data, training_y, epochs=100, callbacks = callbacks, batch_size = 32, validation_data = (validation_data, validation_y))


Score per fold
------------------------------------------------------------------------
> Fold 1 - Loss: 2.6024811267852783 - Accuracy: 0.5740740895271301%
------------------------------------------------------------------------
> Fold 2 - Loss: 4.062525272369385 - Accuracy: 0.5%
------------------------------------------------------------------------
> Fold 3 - Loss: 6.252674102783203 - Accuracy: 0.7037037014961243%
------------------------------------------------------------------------
> Fold 4 - Loss: 2.771653175354004 - Accuracy: 0.6792452931404114%
------------------------------------------------------------------------
> Fold 5 - Loss: 2.699841022491455 - Accuracy: 0.6603773832321167%
------------------------------------------------------------------------
Average scores for all folds:
> Accuracy: 0.6234800934791564 (+- 0.07562444245052322)
> Loss: 3.677834939956665
------------------------------------------------------------------------
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_35 (Dense)             (None, 4096)              78430208  
_________________________________________________________________
dense_36 (Dense)             (None, 1024)              4195328   
_________________________________________________________________
dense_37 (Dense)             (None, 512)               524800    
_________________________________________________________________
dense_38 (Dense)             (None, 128)               65664     
_________________________________________________________________
dense_39 (Dense)             (None, 128)               16512     
_________________________________________________________________
dense_40 (Dense)             (None, 64)                8256      
_________________________________________________________________
dense_41 (Dense)             (None, 2)                 130       
=================================================================



#########
#	2	#
#########
#######################
# batch normalisation
#######################

	model = Sequential()
	model.add(Dense(4096, activation = 'relu', input_dim=19147))
	model.add(BatchNormalization())
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(128))
	model.add(BatchNormalization())
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(2, activation='softmax'))
	# Compile the model
	opt = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC(name='AUC', curve ='ROC'),metrics.AUC(name='PR', curve ='PR')])
	callbacks = [EarlyStopping(monitor='PR', patience=3), ModelCheckpoint(filepath='best_model.h5', monitor='PR', save_best_only=True)]
	print('------------------------------------------------------------------------')
	print(f'Training for fold {fold_var} ...')
	history = model.fit(training_data, training_y, epochs=100, callbacks = callbacks, batch_size = 32, validation_data = (validation_data, validation_y))


Score per fold
------------------------------------------------------------------------
> Fold 1 - Loss: 6.621679306030273 - Accuracy: 0.5370370149612427%
------------------------------------------------------------------------
> Fold 2 - Loss: 4.182056903839111 - Accuracy: 0.4444444477558136%
------------------------------------------------------------------------
> Fold 3 - Loss: 3.5958170890808105 - Accuracy: 0.5925925970077515%
------------------------------------------------------------------------
> Fold 4 - Loss: 2.0570452213287354 - Accuracy: 0.5849056839942932%
------------------------------------------------------------------------
> Fold 5 - Loss: 2.9153811931610107 - Accuracy: 0.5283018946647644%
------------------------------------------------------------------------
Average scores for all folds:
> Accuracy: 0.5374563276767731 (+- 0.052965345055122445)
> Loss: 3.8743959426879884
------------------------------------------------------------------------
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_35 (Dense)             (None, 4096)              78430208  
_________________________________________________________________
batch_normalization_30 (Batc (None, 4096)              16384     
_________________________________________________________________
dense_36 (Dense)             (None, 1024)              4195328   
_________________________________________________________________
batch_normalization_31 (Batc (None, 1024)              4096      
_________________________________________________________________
dense_37 (Dense)             (None, 512)               524800    
_________________________________________________________________
batch_normalization_32 (Batc (None, 512)               2048      
_________________________________________________________________
dense_38 (Dense)             (None, 128)               65664     
_________________________________________________________________
batch_normalization_33 (Batc (None, 128)               512       
_________________________________________________________________
dense_39 (Dense)             (None, 128)               16512     
_________________________________________________________________
batch_normalization_34 (Batc (None, 128)               512       
_________________________________________________________________
dense_40 (Dense)             (None, 64)                8256      
_________________________________________________________________
batch_normalization_35 (Batc (None, 64)                256       
_________________________________________________________________
dense_41 (Dense)             (None, 2)                 130       
=================================================================
Total params: 83,264,706
Trainable params: 83,252,802
Non-trainable params: 11,904
_________________________________________________________________


#########
#	3	#
#########
#####################
# dropout
#####################
https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/


	model = Sequential()
	model.add(Dense(4096, activation = 'relu', input_dim=19147))
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
	print('------------------------------------------------------------------------')
	print(f'Training for fold {fold_var} ...')
	history = model.fit(training_data, training_y, epochs=100, callbacks = callbacks, batch_size = 32, validation_data = (validation_data, validation_y))


Score per fold
------------------------------------------------------------------------
> Fold 1 - Loss: 3.6899049282073975 - Accuracy: 0.5555555820465088%
------------------------------------------------------------------------
> Fold 2 - Loss: 2.539762258529663 - Accuracy: 0.5555555820465088%
------------------------------------------------------------------------
> Fold 3 - Loss: 4.084113121032715 - Accuracy: 0.5185185074806213%
------------------------------------------------------------------------
> Fold 4 - Loss: 1.8773115873336792 - Accuracy: 0.5471698045730591%
------------------------------------------------------------------------
> Fold 5 - Loss: 1.252089023590088 - Accuracy: 0.6226415038108826%
------------------------------------------------------------------------
Average scores for all folds:
> Accuracy: 0.5598881959915161 (+- 0.03420154773464688)
> Loss: 2.6886361837387085
------------------------------------------------------------------------
Model: "sequential_10"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_70 (Dense)             (None, 4096)              78430208  
_________________________________________________________________
batch_normalization_60 (Batc (None, 4096)              16384     
_________________________________________________________________
dense_71 (Dense)             (None, 1024)              4195328   
_________________________________________________________________
batch_normalization_61 (Batc (None, 1024)              4096      
_________________________________________________________________
dropout_12 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_72 (Dense)             (None, 512)               524800    
_________________________________________________________________
batch_normalization_62 (Batc (None, 512)               2048      
_________________________________________________________________
dropout_13 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_73 (Dense)             (None, 128)               65664     
_________________________________________________________________
batch_normalization_63 (Batc (None, 128)               512       
_________________________________________________________________
dense_74 (Dense)             (None, 128)               16512     
_________________________________________________________________
batch_normalization_64 (Batc (None, 128)               512       
_________________________________________________________________
dense_75 (Dense)             (None, 64)                8256      
_________________________________________________________________
batch_normalization_65 (Batc (None, 64)                256       
_________________________________________________________________
dropout_14 (Dropout)         (None, 64)                0         
_________________________________________________________________
dense_76 (Dense)             (None, 2)                 130       
=================================================================
Total params: 83,264,706
Trainable params: 83,252,802
Non-trainable params: 11,904
_________________________________________________________________


#########
#	4	#
#########
#####################
# dropout + maxnorm
#####################
https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

	model = Sequential()
	model.add(Dense(4096, activation = 'relu', input_dim=19147, kernel_constraint=maxnorm(3)))
	model.add(BatchNormalization())
	model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu' ,kernel_constraint=maxnorm(3)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(128, kernel_constraint=maxnorm(3)))
	model.add(BatchNormalization())
	model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	# Compile the model
	opt = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', metrics.AUC(name='AUC', curve ='ROC'),metrics.AUC(name='PR', curve ='PR')])

	callbacks = [EarlyStopping(monitor='PR', patience=3), ModelCheckpoint(filepath='best_model.h5', monitor='PR', save_best_only=True)]
	print('------------------------------------------------------------------------')
	print(f'Training for fold {fold_var} ...')
	history = model.fit(training_data, training_y, epochs=100, callbacks = callbacks, batch_size = 32, validation_data = (validation_data, validation_y))


Score per fold
------------------------------------------------------------------------
> Fold 1 - Loss: 3.868377208709717 - Accuracy: 0.5740740895271301%
------------------------------------------------------------------------
> Fold 2 - Loss: 3.8221797943115234 - Accuracy: 0.5185185074806213%
------------------------------------------------------------------------
> Fold 3 - Loss: 6.221246242523193 - Accuracy: 0.5370370149612427%
------------------------------------------------------------------------
> Fold 4 - Loss: 4.510375499725342 - Accuracy: 0.49056604504585266%
------------------------------------------------------------------------
> Fold 5 - Loss: 6.423251152038574 - Accuracy: 0.5094339847564697%
------------------------------------------------------------------------
Average scores for all folds:
> Accuracy: 0.5259259283542633 (+- 0.0283509333360743)
> Loss: 4.96908597946167
------------------------------------------------------------------------
Model: "sequential_15"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_105 (Dense)            (None, 4096)              78430208  
_________________________________________________________________
batch_normalization_90 (Batc (None, 4096)              16384     
_________________________________________________________________
dense_106 (Dense)            (None, 1024)              4195328   
_________________________________________________________________
batch_normalization_91 (Batc (None, 1024)              4096      
_________________________________________________________________
dropout_27 (Dropout)         (None, 1024)              0         
_________________________________________________________________
dense_107 (Dense)            (None, 512)               524800    
_________________________________________________________________
batch_normalization_92 (Batc (None, 512)               2048      
_________________________________________________________________
dropout_28 (Dropout)         (None, 512)               0         
_________________________________________________________________
dense_108 (Dense)            (None, 128)               65664     
_________________________________________________________________
batch_normalization_93 (Batc (None, 128)               512       
_________________________________________________________________
dense_109 (Dense)            (None, 128)               16512     
_________________________________________________________________
batch_normalization_94 (Batc (None, 128)               512       
_________________________________________________________________
dense_110 (Dense)            (None, 64)                8256      
_________________________________________________________________
batch_normalization_95 (Batc (None, 64)                256       
_________________________________________________________________
dropout_29 (Dropout)         (None, 64)                0         
_________________________________________________________________
dense_111 (Dense)            (None, 2)                 130       
=================================================================
Total params: 83,264,706
Trainable params: 83,252,802
Non-trainable params: 11,904
_________________________________________________________________

