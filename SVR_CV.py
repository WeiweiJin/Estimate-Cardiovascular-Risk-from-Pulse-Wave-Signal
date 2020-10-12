#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:32:58 2020

@author: weiweijin
"""

# %% Importing the libraries 
import pandas as pd
import sys
import optunity
import optunity.metrics

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from itertools import compress

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

# we explicitly generate the outer_cv decorator so we can use it twice
outer_cv = optunity.cross_validated(x=S_feature, y=target, num_folds=3)

# Baseline SVM model (RBF)
def compute_mse_standard(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and default hyperparameters.
    """
    model = SVR(gamma = 'auto').fit(x_train, y_train)
    predictions = model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)

# wrap with outer cross-validation
compute_mse_standard = outer_cv(compute_mse_standard)

baseline_std = compute_mse_standard()

# %% CV for SVM
# start CV
print('start CV')

# saving the paramters
orig_stdout = sys.stdout
f = open('outputLassoSVR_CV.txt', 'w')
sys.stdout = f

def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=10, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, C):
        model = SVR(gamma = 'auto', C=C, kernel='rbf').fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[10, 200])
    print("optimal hyperparameters: " + str(optimal_pars))

    tuned_model = SVR(kernel='rbf', gamma = 'auto', **optimal_pars).fit(x_train, y_train)
    predictions = tuned_model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)

# wrap with outer cross-validation
compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned)

std_tuned = compute_mse_rbf_tuned()


# %% save report

print('baseline std=', baseline_std)
print('Tuned std (RBF) = ', std_tuned)

sys.stdout = orig_stdout
f.close()