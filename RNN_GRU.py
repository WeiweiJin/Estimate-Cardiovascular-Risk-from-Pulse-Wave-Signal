#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:08:25 2020

@author: weiweijin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys
import os
import math
import datetime
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from bland_altman_plot import bland_altman_plot


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
target_mat = target_mat.astype('float32')

#split traning and testing datasets
X_train,X_test,y_train,y_test=train_test_split(sub_mat,target_mat, test_size=0.3, random_state=31)

# %% RNN (GRU)
tf.keras.backend.set_floatx('float32') # change data type to float 64

timesteps =  sub_mat.shape[1]
features =  1

# data pipeline
batch_size = 128
buffer_size = 250
epochs = 1500
steps_per_epoch = math.floor(X_train.shape[0] / batch_size)
validation_steps = math.floor(X_test.shape[0] / batch_size)

traData = tf.data.Dataset.from_tensor_slices((X_train, y_train))
traData = traData.cache().shuffle(buffer_size).batch(batch_size).repeat()

testData = tf.data.Dataset.from_tensor_slices((X_test, y_test))
testData = testData.batch(batch_size).repeat()

gru_model = tf.keras.models.Sequential()
gru_model.add(layers.Masking(mask_value = np.finfo('float32').max, input_shape=(timesteps, features)))
gru_model.add(layers.Bidirectional(layers.GRU(16)))
gru_model.add(layers.Dense(1))
gru_model.compile(optimizer = 'adam', loss = 'mse', metrics=['RootMeanSquaredError'])

# traning the model
gru_model.summary()

# creat tensorboard
NAME = "Test{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tboard_log_dir = os.path.join("logs",NAME)
tensorboard = TensorBoard(log_dir = tboard_log_dir)

training_history = gru_model.fit(traData, epochs=epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=testData, validation_steps=validation_steps,
                      callbacks=tensorboard)


print("Average test loss: ", np.average(training_history.history['loss']))

y_pred = gru_model.predict(X_test)


# %% Save printouts
orig_stdout = sys.stdout
f = open('output_GRU_RNN.txt', 'w')
sys.stdout = f

print('Mean Absolute Error(Validation/Testing):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error(Validation/Testing):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error(Validation/Testing):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

sys.stdout = orig_stdout
f.close()

# Plot fig
# blandAltman plot
fig1 = plt.figure(figsize = (8,8))
ax1 = fig1.add_subplot(1,1,1)
ax1 = bland_altman_plot(y_test, y_pred)
plt.ylim(-5,6)
ax1.tick_params(axis='x',labelsize = 20)
ax1.tick_params(axis='y',labelsize = 20)
plt.xticks(np.arange(6, 19, 6)) 
plt.yticks(np.arange(-4, 5, 4)) 
filename1 = 'GRU_Bland_Altman.png'
fig1.savefig(filename1)

# Estimated vs measured
m, b = np.polyfit(y_pred.reshape(-1), y_test.reshape(-1),1)
X = sm.add_constant(y_pred)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]
r_squared = est2.rsquared
fig2 = plt.figure(figsize = (8,8))
ax2 = fig2.add_subplot(1,1,1)
plt.plot(y_pred, y_test, 'k.', markersize = 10)
ax2.plot(y_pred, m*y_pred +b, 'r', label = 'y = {:.2f}x+{:.2f}'.format(m, b))
plt.xlim(3,18)
plt.ylim(3,25)
ax2.tick_params(axis='x',labelsize = 20)
ax2.tick_params(axis='y',labelsize = 20)
plt.xticks(np.arange(3, 16, 6)) 
plt.yticks(np.arange(3, 22, 6))
plt.legend(fontsize=18,loc=2)
ax2.text(4, 21, 'r$^2$ = {:.2f}'.format(r_squared), fontsize=18)
ax2.text(4, 19, 'p < 0.0001', fontsize=18)
filename2 = 'GRU_est_vs_med.png'
fig2.savefig(filename2) 

# save data
GRU_result = pd.DataFrame(data = y_test, columns = ['y_test'])
GRU_result2 = pd.DataFrame(data = y_pred, columns = ['y_pred'])
GRU_result = pd.concat([GRU_result, GRU_result2], axis = 1)
GRU_result.to_pickle('GRU_output_data.pkl')

