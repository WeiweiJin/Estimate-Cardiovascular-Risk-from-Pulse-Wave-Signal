#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:44:14 2020

@author: weiweijin
"""

# %% Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# %% Import data
Twin_data = pd.read_pickle("processed_data2.pkl")

# %% Preparing data for main analysis
# get rid of the excess data
del_name = ['Visit_No', 'Height', 'Weight', 'BMI', 'CSBP', 'CDBP','Age','AP',
            'AI','DBP','SBP','MAP','ms2_v','ms2_t', 'RI', 'AGI_mod','t_dia','CT','delta_t','Sex']
Twin_data = Twin_data.drop(columns = del_name)

#get rid of nans 
Twin_data = Twin_data.apply(pd.to_numeric, errors = 'coerce')
Twin_data = Twin_data.dropna()
Twin_data.reset_index(drop = True, inplace = True)

# feature name 
target_name = ['PWV']
feature_name = list(Twin_data.columns)
feature_name.remove('PWV')

# Separating out the features
x = Twin_data.loc[:, feature_name].values

# Separating out the target
y = Twin_data.loc[:,target_name].values

# Standardizing the features
X = StandardScaler().fit_transform(x)

# %% Lasso CV
lasso = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

# %% plot error lines showing +/- std. errors of the scores
fig = plt.figure(figsize = (8,8))
plt.semilogx(alphas, scores)

std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

plt.ylabel('CV score +/- std error', fontsize = 20)
plt.xlabel('alpha', fontsize = 20)
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])
fig.savefig('lassoCV.pdf')
