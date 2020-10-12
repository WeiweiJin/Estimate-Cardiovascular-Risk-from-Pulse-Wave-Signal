#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:43:13 2020

@author: weiweijin
"""

# %% Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
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
f = open('outputGBoost.txt', 'w')
sys.stdout = f

# %% CV using random search
# Loss function
loss = ['ls', 'lad', 'huber']
# Learning rate
learning_rate = [0.01, 0.02, 0.05, 0.1]
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 10, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [40, 60, 80]
# Minimum number of samples required at each leaf node
min_samples_leaf = [20, 30, 40]

# Create the random grid
random_grid = {'loss': loss, 
               'learning_rate': learning_rate,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               }
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
gb = GradientBoostingRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gb_GBR = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
gb_GBR.fit(x, y)
# show best parameter
gb_GBR.best_params_

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

base_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
base_model.fit(x, y)
base_accuracy = evaluate(base_model, x, y)

best_GBR = gb_GBR.best_estimator_
GBR_accuracy = evaluate(best_GBR, x, y)

print('Improvement(Random grid) of {:0.2f}%.'.format( 100 * (GBR_accuracy - base_accuracy) / base_accuracy))


# CV error matrix
GB_rs = GradientBoostingRegressor(loss=best_GBR.loss, 
                              learning_rate = best_GBR.learning_rate,
                              max_features = best_GBR.max_features,
                              max_depth = best_GBR.max_depth,
                              min_samples_split = best_GBR.min_samples_split,
                              min_samples_leaf = best_GBR.min_samples_leaf,
                              n_estimators=best_GBR.n_estimators, 
                              random_state=False, verbose=False)
pred_rs = cross_val_predict(GB_rs, x, y, cv=10)
print('Mean Absolute Error(Random search):', metrics.mean_absolute_error(y, pred_rs))
print('Mean Squared Error(Random search):', metrics.mean_squared_error(y, pred_rs))
print('Root Mean Squared Error(Random search):', np.sqrt(metrics.mean_squared_error(y, pred_rs)))

# predict with the best parameters from random search
y_test = y_test.reshape(-1)
GB_rs.fit(X_train,y_train)
y_pred_rs = GB_rs.predict(X_test)

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
filename1 = 'GBoost_Bland_Altman.png'
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
filename2 = 'GBoost_est_vs_med.png'
fig2.savefig(filename2) 

# save target_test and target_pred
savedata = [y_test, y_pred_rs]
df_savedata = pd.DataFrame(savedata)
df_savedata.to_pickle('y_test_pred_GBoost.pkl')


# Get numerical feature importances
importances_rs = list(GB_rs.feature_importances_)

# List of tuples with variable and importance
feature_importances_rs = [(Twin_data, round(importance_rs, 2)) for Twin_data, importance_rs in zip(feature_name, importances_rs)]

# Sort the feature importances by most important first
feature_importances_rs = sorted(feature_importances_rs, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_rs];

# Save feature importances
F_Imp_rs = pd.DataFrame(feature_importances_rs, columns =['FeatureName','Importance'])
F_Imp_rs.to_csv('Importances_GBoost_RS.csv')

# # %% CV using Grid Search
# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'loss': ['ls', 'lad', 'huber'], 
#     'learning_rate': [0.01, 0.02, 0.05, 0.1],
#     'max_depth': [3, 5, 10, 12],
#     'max_features': [2, 3],
#     'min_samples_leaf': [20, 30, 40],
#     'min_samples_split': [40, 60, 80],
#     'n_estimators': [100, 200, 300, 500]
# }
# # Create a based model
# gb = GradientBoostingRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = gb, param_grid = param_grid, 
#                           cv = 10, n_jobs = -1, verbose = 2)
# # Fit the grid search to the data
# grid_result = grid_search.fit(x, y)
# best_params = grid_search.best_params_

# best_grid = grid_search.best_estimator_
# grid_accuracy = evaluate(best_grid, x, y)

# print('Improvement(Grid search) of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

# # CV error matrix
# GB_gs = GradientBoostingRegressor(loss=best_params["loss"],learning_rate=best_params["learning_rate"],
#                                 max_depth=best_params["max_depth"], max_features = best_params["max_features"],
#                                 min_samples_split = best_params["min_samples_split"],
#                                 min_samples_leaf = best_params["min_samples_leaf"],
#                                 n_estimators=best_params["n_estimators"], random_state=False, verbose=False)
# pred_grd = cross_val_predict(GB_gs, x, y, cv=10)

# #Evaluating the algorithm
# print('Mean Absolute Error(Grid search):', metrics.mean_absolute_error(y, pred_grd))
# print('Mean Squared Error(Grid search):', metrics.mean_squared_error(y, pred_grd))
# print('Root Mean Squared Error(Grid search):', np.sqrt(metrics.mean_squared_error(y, pred_grd)))

# # predict with the best parameters from grid search
# y_test = y_test.reshape(-1)
# GB_gs.fit(X_train,y_train)
# y_pred_gs = GB_gs.predict(X_test)
# # Plot fig
# fig = plt.figure(figsize = (8,10))
# ax = fig.add_subplot(2,1,1) 
# plt.plot(y_test, 'r.', markersize=10)
# plt.plot(y_pred_gs, 'b*', markersize=10)
# plt.ylim(0,25)
# ax.tick_params(axis='x',labelsize = 16)
# ax.tick_params(axis='y',labelsize = 16)
# ax.set_xlabel('Data sample No.', fontsize = 20)
# ax.set_ylabel('PWV (m/s)', fontsize = 20)
# plt.xticks(np.arange(0, 1010, 500)) 
# plt.yticks(np.arange(0, 22, 10)) 
# ax.legend(['Observations', 'Prediction'],fontsize=18)     #, '95% confidence interval'
# # blandAltman plot
# ax2 = fig.add_subplot(2,1,2)
# sm.graphics.mean_diff_plot(y_test, y_pred_gs, ax = ax2)
# plt.ylim(-5,6)
# #ax2 = bland_altman_plot(y_test, y_pred_gs)
# ax2.tick_params(axis='x',labelsize = 16)
# ax2.tick_params(axis='y',labelsize = 16)
# plt.xticks(np.arange(6, 19, 6)) 
# plt.yticks(np.arange(-4, 5, 4))
# ax2.set_xlabel('Mean PWV (m/s)', fontsize = 20)
# ax2.set_ylabel('Difference (m/s)', fontsize = 20)
# filename = 'GB_Grid_search_pred.png'
# fig.savefig(filename)

# # Get numerical feature importances
# importances_gs = list(GB_gs.feature_importances_)

# # List of tuples with variable and importance
# feature_importances_gs = [(Twin_data, round(importance_gs, 2)) for Twin_data, importance_gs in zip(feature_name, importances_gs)]

# # Sort the feature importances by most important first
# feature_importances_gs = sorted(feature_importances_gs, key = lambda x: x[1], reverse = True)

# # Print out the feature and importances 
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_gs];

# # Save feature importances
# F_Imp_gs = pd.DataFrame(feature_importances_gs, columns =['FeatureName','Importance'])
# F_Imp_gs.to_csv('Importances_GBoost_GS.csv')


# # %% Save the model
# import pickle
# filename = 'GB_model_rs.sav'
# pickle.dump(GB_rs, open(filename, 'wb'))
# filename2 = 'GB_model_gs.sav'
# pickle.dump(GB_gs, open(filename2, 'wb'))

# %% close the output file
sys.stdout = orig_stdout
f.close()
