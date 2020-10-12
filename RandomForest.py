#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:38:19 2020

@author: weiweijin
"""

# %% Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from pprint import pprint
from bland_altman_plot import bland_altman_plot

# %% Import data
Twin_data = pd.read_pickle("Twin_data_no_outliner.pkl")

# %% Preparing data for main analysis
# feature name 
target_name = ['PWV']
feature_name = list(Twin_data.columns)
feature_name.remove('PWV')

# Separating out the features
x = Twin_data.loc[:, feature_name].values

# Separating out the target
y = Twin_data.loc[:,target_name].values
y = y.reshape(-1)

#split traning and testing datasets
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=31)

# %% Save printouts
orig_stdout = sys.stdout
f = open('outputRF.txt', 'w')
sys.stdout = f

# %% CV using random search
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 20, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [40, 60, 80]
# Minimum number of samples required at each leaf node
min_samples_leaf = [20, 30, 40]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x, y)
# show best parameter
rf_random.best_params_

#Evaluate random search
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance(Random grid)')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestRegressor(n_estimators=100, random_state=0)
base_model.fit(x, y)
base_accuracy = evaluate(base_model, x, y)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x, y)

print('Improvement(Random grid) of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# CV error matrix
rf_rs = RandomForestRegressor(max_depth=best_random.max_depth, 
                              max_features = best_random.max_features,
                              min_samples_split = best_random.min_samples_split,
                              min_samples_leaf = best_random.min_samples_leaf,
                              n_estimators=best_random.n_estimators, 
                              bootstrap = best_random.bootstrap,
                              random_state=False, verbose=False)
pred_rs = cross_val_predict(rf_rs, x, y, cv=10)
print('Mean Absolute Error(Random search):', metrics.mean_absolute_error(y, pred_rs))
print('Mean Squared Error(Random search):', metrics.mean_squared_error(y, pred_rs))
print('Root Mean Squared Error(Random search):', np.sqrt(metrics.mean_squared_error(y, pred_rs)))

# predict with the best parameters from random search
y_test = y_test.reshape(-1)
rf_rs.fit(X_train,y_train)
y_pred_rs = rf_rs.predict(X_test)

# Plot fig
# blandAltman plot
fig1 = plt.figure(figsize = (8,8))
ax1 = fig1.add_subplot(1,1,1)
ax1 = bland_altman_plot(y_test, y_pred_rs)
plt.ylim(-5,6)
ax1.tick_params(axis='x',labelsize = 20)
ax1.tick_params(axis='y',labelsize = 20)
plt.xticks(np.arange(6, 19, 6)) 
plt.yticks(np.arange(-4, 5, 4)) 
filename1 = 'RF_Bland_Altman.png'
fig1.savefig(filename1)

# Estimated vs measured
m, b = np.polyfit(y_pred_rs, y_test,1)
X = sm.add_constant(y_pred_rs)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]
r_squared = est2.rsquared
fig2 = plt.figure(figsize = (8,8))
ax2 = fig2.add_subplot(1,1,1)
plt.plot(y_pred_rs, y_test, 'k.', markersize = 10)
ax2.plot(y_pred_rs, m*y_pred_rs +b, 'r', label = 'y = {:.2f}x+{:.2f}'.format(m, b))
plt.xlim(3,18)
plt.ylim(3,25)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.xticks(np.arange(3, 16, 6)) 
plt.yticks(np.arange(3, 22, 6))
plt.legend(fontsize=18,loc=2)
ax2.text(4, 21, 'r$^2$ = {:.2f}'.format(r_squared), fontsize=18)
ax2.text(4, 19, 'p < 0.0001', fontsize=18)
filename2 = 'RF_est_vs_med.png'
fig2.savefig(filename2) 

# save target_test and target_pred
savedata = [y_test, y_pred_rs]
df_savedata = pd.DataFrame(savedata)
df_savedata.to_pickle('y_test_pred_RF.pkl')

# Get numerical feature importances
importances_rs = list(rf_rs.feature_importances_)

# List of tuples with variable and importance
feature_importances_rs = [(Twin_data, round(importance_rs, 2)) for Twin_data, importance_rs in zip(feature_name, importances_rs)]

# Sort the feature importances by most important first
feature_importances_rs = sorted(feature_importances_rs, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_rs];

# Save feature importances
F_Imp_rs = pd.DataFrame(feature_importances_rs, columns =['FeatureName','Importance'])
F_Imp_rs.to_csv('Importances_RF_RS.csv')


# %% close the output file
sys.stdout = orig_stdout
f.close()

