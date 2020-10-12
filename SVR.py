#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:26:08 2020

@author: weiweijin
"""

# %% Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import statsmodels.api as sm
import sys

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from itertools import compress
from sklearn import metrics
from bland_altman_plot import bland_altman_plot

# %% Import data
Twin_data = pd.read_pickle("Twin_data_no_outliner.pkl")

# %% Preparing data for main analysis
# feature name 
target_name = ['PWV']
feature_name = list(Twin_data.columns)
feature_name.remove('PWV')

# %% Lasso
# Separating out the features
x = Twin_data.loc[:, feature_name].values

# Separating out the target
y = Twin_data.loc[:,target_name].values
y = y.reshape(-1)

# Standardizing the features
XL = StandardScaler().fit_transform(x)

#perform Lasso regreession
lasso_tuned = Lasso(alpha=0.01, max_iter=10e5)
lasso_tuned.fit(XL,y)

# %% prepare data for SVM
# evaluate the seperated features
LassoKInd = abs(lasso_tuned.coef_) > 0.001
feature_name_SVR = list(compress(feature_name,LassoKInd))
SVR_data = Twin_data[feature_name_SVR]

# Separating out the features
SVR_features = SVR_data.loc[:, feature_name_SVR].values
# Separateing out the targets
target = Twin_data.loc[:, target_name].values
target = target.reshape(-1)
# Standardizing the features
S_feature = StandardScaler().fit_transform(SVR_features)

# %% Predict using the tuned parameters 
#split traning and testing datasets
Feature_train,Feature_test,Target_train,Target_test = train_test_split(S_feature,target, test_size=0.3, random_state=31)

# SVM model
SVM_tuned = SVR(kernel='rbf', gamma = 'auto', C=10.113535298901722)
# Fit the tuned model 
SVM_tuned.fit(Feature_train, Target_train)
# Predict the target
Target_pred = SVM_tuned.predict(Feature_test)

# Plot fig
# blandAltman plot
fig1 = plt.figure(figsize = (8,8))
ax1 = fig1.add_subplot(1,1,1)
ax1 = bland_altman_plot(Target_test, Target_pred)
plt.ylim(-5,6)
ax1.tick_params(axis='x',labelsize = 20)
ax1.tick_params(axis='y',labelsize = 20)
plt.xticks(np.arange(6, 19, 6)) 
plt.yticks(np.arange(-4, 5, 4)) 
filename1 = 'SVR_Bland_Altman.png'
fig1.savefig(filename1)

# Estimated vs measured
m, b = np.polyfit(Target_pred, Target_test,1)
X = sm.add_constant(Target_pred)
est = sm.OLS(Target_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]
r_squared = est2.rsquared
fig2 = plt.figure(figsize = (8,8))
ax2 = fig2.add_subplot(1,1,1)
plt.plot(Target_pred, Target_test, 'k.', markersize = 10)
ax2.plot(Target_pred, m*Target_pred +b, 'r', label = 'y = {:.2f}x+{:.2f}'.format(m, b))
plt.xlim(3,18)
plt.ylim(3,25)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.xticks(np.arange(3, 16, 6)) 
plt.yticks(np.arange(3, 22, 6))
plt.legend(fontsize=18,loc=2)
ax2.text(4, 21, 'r$^2$ = {:.2f}'.format(r_squared), fontsize=18)
ax2.text(4, 19, 'p < 0.0001', fontsize=18)
filename2 = 'SVR_est_vs_med.png'
fig2.savefig(filename2) 

# save target_test and target_pred
savedata = [Target_test, Target_pred]
df_savedata = pd.DataFrame(savedata)
df_savedata.to_pickle('y_test_pred_SVR.pkl')

# %% save report
orig_stdout = sys.stdout
f = open('outputLassoSVR.txt', 'w')
sys.stdout = f

#Evaluating the tuned SVM model
print('Mean Absolute Error:', metrics.mean_absolute_error(Target_test, Target_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Target_test, Target_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Target_test, Target_pred)))   

sys.stdout = orig_stdout
f.close()