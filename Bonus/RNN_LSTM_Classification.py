#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:08:25 2020

@author: weiweijin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import math
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

# %% normalization function using the standard range of BP (i.e. min = 60mmHg, max = 120 mmHg)
def norm_to_zero_one(df):
    return (df - 60) * 1.0 / 60

# %% import data
# import waveform data 
Twin_data = pd.read_pickle("processed_PW.pkl")
del_col_names = ['Visit_No','Weight', 'BMI', 'SBP', 'Sex', 'Age', 'Height', 'DBP', 'MAP',
                 'CSBP', 'CDBP', 'HR']
PW_data = Twin_data.drop(columns = del_col_names)
PW_data = PW_data[PW_data['PWV'].notna()]
PW_data.reset_index(drop = True, inplace = True)
target_name = ['PWV']
target = PW_data[target_name]
target.loc[(target['PWV']<=10.0)] = 0
target.loc[(target['PWV']>10.0)] = 1
# extract PW data
PW = PW_data.drop(columns = 'PWV')
# normalise the pressure wave by using the standard range of BP (i.e. min = 60mmHg, max = 120 mmHg)
sub_norm = PW.apply(norm_to_zero_one)
sub_norm = sub_norm.fillna(np.finfo('float32').max)

# %% data preparation
# input data matrix
sub_norm = sub_norm.T
subject_no = list(sub_norm.columns)
sub_mat = sub_norm.loc[:, subject_no].values
sub_mat = np.transpose(sub_mat)
sub_mat = np.expand_dims(sub_mat,axis = -1)
sub_mat = sub_mat.astype('float32')

# output data array
target_mat = target.loc[:,target_name].values
target_mat = target_mat.astype('int32')

#split traning and testing datasets
X_train,X_test,y_train,y_test=train_test_split(sub_mat,target_mat, test_size=0.3, random_state=31)

# %% RNN (LSTM)
tf.keras.backend.set_floatx('float32') # change data type to float 64

timesteps =  sub_mat.shape[1]
features =  1

# data pipeline
batch_size = 64
buffer_size = 250
epochs = 15
steps_per_epoch = math.floor(X_train.shape[0] / batch_size)
validation_steps = math.floor(X_test.shape[0] / batch_size)

traData = tf.data.Dataset.from_tensor_slices((X_train, y_train))
traData = traData.cache().shuffle(buffer_size).batch(batch_size).repeat()

testData = tf.data.Dataset.from_tensor_slices((X_test, y_test))
testData = testData.batch(batch_size).repeat()

lstm_model = tf.keras.models.Sequential()
lstm_model.add(layers.Masking(mask_value = np.finfo('float32').max, input_shape=(timesteps, features)))
# lstm_model.add(layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
lstm_model.add(layers.Bidirectional(layers.LSTM(16)))
lstm_model.add(layers.Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['BinaryAccuracy'])

# traning the model
lstm_model.summary()

# creat tensorboard
NAME = "Test{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tboard_log_dir = os.path.join("logs",NAME)
tensorboard = TensorBoard(log_dir = tboard_log_dir)

training_history = lstm_model.fit(traData, epochs=epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=testData, validation_steps=validation_steps,
                      callbacks=tensorboard)

print("Average test loss: ", np.average(training_history.history['loss']))

y_pred = lstm_model.predict(X_test).astype('int32')

# %% Postprocess
# evaluate using ROC curve
# calcuate probablities for the class 0, let's assume it is the healthy patients
y_probs = lstm_model.predict_proba(X_test)
# calculate roc auc
auc = roc_auc_score(y_test, y_probs) 
# calcuate roc curve
fpr, tpr, _ = roc_curve(y_test, y_probs) 
#plot
fig2 = plt.figure(figsize = (8,6))
ax = fig2.add_subplot(1,1,1) 
plt.plot(fpr, tpr, 'b-', linewidth=2)
ax.tick_params(axis='x',labelsize = 20)
ax.tick_params(axis='y',labelsize = 20)
plt.xlabel('False Positive Rate', fontsize = 24)
plt.ylabel('True Positive Rate', fontsize = 24)
ax.set_title("ROC AUC = %.3f" % auc ,fontsize = 24);
filename = 'ROC.png'
fig2.savefig(filename)

# evaluate using confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
classes = ['Low', 'High']
fig3= plt.figure(figsize = (5, 5))
ax3 = fig3.add_subplot(1,1,1) 
ax3.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Oranges)
ax3.set_title('Pulse Wave Velocity', size = 24)
ax3.set_aspect('equal', 'box')

tick_marks = np.arange(len(classes))

ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)

ax3.set_xlim([-0.5, 1.5])
ax3.set_ylim([1.5, -0.5])
ax3.set_xticklabels(classes, fontsize = 16)
ax3.set_yticklabels(classes, fontsize = 16, rotation=90)

# save data
LSTM_result = pd.DataFrame(data = y_test, columns = ['y_test'])
LSTM_result2 = pd.DataFrame(data = y_pred, columns = ['y_pred'])
LSTM_result = pd.concat([LSTM_result, LSTM_result2], axis = 1)
LSTM_result.to_pickle('LSTM_output_data.pkl')
