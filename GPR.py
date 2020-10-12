#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:57:05 2020

@author: weiweijin
"""

# %% Importing the libraries 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, Matern
from itertools import compress
from sklearn import metrics
from bland_altman_plot import bland_altman_plot

# %% Import data
Twin_data = pd.read_pickle("Twin_data_no_outliner.pkl")

# %% Preparing data for main analysis
# get target and feature name 
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

# %% Prepare data for GP 
# evaluate the seperated features
LassoKInd = abs(lasso_tuned.coef_) > 0.001
feature_name_SVM = list(compress(feature_name,LassoKInd))
GP_data = Twin_data[feature_name_SVM]

# Separating out the features
GP_features = GP_data.loc[:, feature_name_SVM].values
# Separateing out the targets
target = Twin_data.loc[:, target_name].values
target = target.reshape(-1)
# Standardizing the features
S_feature = StandardScaler().fit_transform(GP_features)

#split traning and testing datasets
X_train,X_test,y_train,y_test=train_test_split(S_feature,target, test_size=0.3, random_state=31)

# %% Perform GP with different kernels
# define different kernels
kernels = [1.0 * RBF(1, (1, 1)),
           1.0 * Matern(2, (2, 2),2.5),
           1.0 * RationalQuadratic(0.7, 0.5, (0.5, 1), (0.5,0.6)),
           1.0 * RBF(1, (1, 1)) + 1.0 * Matern(1/0.5, (1/0.5, 1/0.5),2.5),
           1.0 * RationalQuadratic(0.7, 0.5, (0.5, 1), (0.5,0.6)) 
           + 1.0 * Matern(2, (2, 2),2.5),
           1.0 * RationalQuadratic(0.7, 0.5, (0.5, 1), (0.5,0.6)) 
           + 1.0 * RBF(1, (1, 1)),
           1.0 * RationalQuadratic(0.7, 0.5, (0.5, 1), (0.5,0.6)) 
           + 1.0 * RBF(1, (1, 1)) + 1.0 * Matern(2, (2, 2),2.5)]

# Save printouts
orig_stdout = sys.stdout
f = open('outputLassoGP.txt', 'w')
ii = 1
sys.stdout = f

# list containing test and predict data
savedata = [y_test]

# run GP and save the plots
for kernel in kernels:
    # Specify Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel)
    model = gp.fit(X_train, y_train)
    s = gp.score(X_train, y_train) #R2 of the prediction
    y_pred, sigma = gp.predict(X_test, return_std=True)
    #Evaluating the algorithm
    print('Mean Absolute Error(', gp.kernel_, '):', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error(', gp.kernel_, '):', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error(', gp.kernel_, '):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    # add y_pred to the list
    savedata.append(y_pred)
    
    # Plot fig
    fig1 = plt.figure(figsize = (8,8))
    ax1 = fig1.add_subplot(1,1,1) 
    plt.plot(y_test, 'r.', markersize=10)
    xn = np.arange(0,y_pred.size, 1)
    plt.errorbar(xn, y_pred, yerr = sigma*1.96, fmt='b*', markersize=10)
    plt.ylim(0,25)
    ax1.tick_params(axis='x',labelsize = 20)
    ax1.tick_params(axis='y',labelsize = 20)
    plt.xticks(np.arange(0, 1010, 500)) 
    plt.yticks(np.arange(0, 22, 10)) 
    filename1 = 'GP_RQ'+ str(ii) + '.png'
    fig1.savefig(filename1)
    
    # Zoom in view
    fig2 = plt.figure(figsize = (8,8))
    ax2 = fig2.add_subplot(1,1,1)
    plt.plot(y_test[0:10], 'r.', markersize=10)
    xn2 = np.arange(0,10, 1)
    plt.errorbar(xn2, y_pred[0:10], yerr = sigma[0:10]*1.96, fmt='b*', markersize=10)
    plt.ylim(0,25)
    ax2.tick_params(axis='x',labelsize = 20)
    ax2.tick_params(axis='y',labelsize = 20)
    plt.xticks(np.arange(0, 12, 5))
    plt.yticks(np.arange(0, 22, 10)) 
    ax2.legend(['Measured', 'Estimated (with 95% CI)'],fontsize=18)     #, '95% confidence interval'
    filename2 = 'GP_RQ_zoom'+ str(ii) + '.png'
    fig2.savefig(filename2)

    # blandAltman plot
    fig3 = plt.figure(figsize = (8,8))
    ax3 = fig3.add_subplot(1,1,1)
    ax3 = bland_altman_plot(y_test, y_pred)
    plt.ylim(-5,6)
    ax3.tick_params(axis='x',labelsize = 20)
    ax3.tick_params(axis='y',labelsize = 20)
    plt.xticks(np.arange(6, 19, 6)) 
    plt.yticks(np.arange(-4, 5, 4)) 
    filename3 = 'GP_RQ_Bland_Altman'+ str(ii) + '.png'
    fig3.savefig(filename3)

    # Estimated vs measured
    m, b = np.polyfit(y_pred, y_test,1)
    X = sm.add_constant(y_pred)
    est = sm.OLS(y_test, X)
    est2 = est.fit()
    p_value =  est2.pvalues[1]
    r_squared = est2.rsquared
    fig4 = plt.figure(figsize = (8,8))
    ax4 = fig4.add_subplot(1,1,1)
    plt.plot(y_pred, y_test, 'k.', markersize = 10)
    ax4.plot(y_pred, m*y_pred +b, 'r', label = 'y = {:.2f}x+{:.2f}'.format(m, b))
    plt.xlim(3,18)
    plt.ylim(3,25)
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.xticks(np.arange(3, 16, 6)) 
    plt.yticks(np.arange(3, 22, 6))
    plt.legend(fontsize=18,loc=2)
    ax4.text(4, 21, 'r$^2$ = {:.2f}'.format(r_squared), fontsize=18)
    ax4.text(4, 19, 'p < 0.0001', fontsize=18)
    filename4 = 'GP_RQ_est_vs_med'+ str(ii) + '.png'
    fig4.savefig(filename4) 
    
    ii = ii + 1
# close the output file
sys.stdout = orig_stdout
f.close()

# save y_test and y_pred
df_savedata = pd.DataFrame(savedata)
df_savedata.to_pickle('y_test_pred_GPR.pkl')
