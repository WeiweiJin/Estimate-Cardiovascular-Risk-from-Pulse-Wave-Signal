#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:47:13 2020

@author: weiweijin
"""

# %% Importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import compress

# %% functions for finding the outliers
def func2DPCA(tol, target, finalDf, PCs):
    # plot 2-D PCA figure
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.xlim(-12,14)
    plt.ylim(-6,8)
    ax.set_xlabel('PC 1 ('+str(round(PCs[0]*100,2))+'%)', fontsize = 20)
    ax.set_ylabel('PC 2 ('+str(round(PCs[1]*100,2))+'%)', fontsize = 20)
    ax.set_title('Grouped by age', fontsize = 20)

    targetcol = [target < 7,np.logical_and(target >= 7, target <= 9), target > 9]
    colors = ['r', 'b', 'g']

    for targetK, color in zip(targetcol,colors):
        indicesToKeep = targetK
        ax.scatter(finalDf.loc[indicesToKeep.to_numpy().reshape(-1), 'PC1']
                   , finalDf.loc[indicesToKeep.to_numpy().reshape(-1), 'PC2']
                   , c = color
                   , s = 20) 

    plt.xticks(np.arange(-12, 14, 12)) 
    plt.yticks(np.arange(-6, 8, 6))      
    ax.legend(['<7', '7-9','>9'], fontsize=18)    
    plt.show()
            
    fig.savefig('lassoPCA.pdf')
    
    # identifiy the outliners
    L = len(target)
    outliner = []
    for i in range(L):
        dis = finalDf['PC1'][i]**2 + finalDf['PC2'][i]**2
        if dis > tol**2:
            outliner = np.append(outliner, i)

    #PCA plot without outliners
    finalDf_new = finalDf.drop(outliner)
    finalDf_new.reset_index(drop = True, inplace = True)
    target_new = target.drop(outliner)
    target_new.reset_index(drop = True, inplace = True)            
    fig2 = plt.figure(figsize = (8,8))
    ax2 = fig2.add_subplot(1,1,1) 
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.xlim(-12,14)
    plt.ylim(-6,8)
    ax2.set_xlabel('PC 1 ('+str(round(PCs[0]*100,2))+'%)', fontsize = 20)
    ax2.set_ylabel('PC 2 ('+str(round(PCs[1]*100,2))+'%)', fontsize = 20)
    ax2.set_title('Grouped by age', fontsize = 20)

    targetcol_new = [target_new < 7,np.logical_and(target_new >= 7, target_new <= 9), target > 9]

    for targetK, color in zip(targetcol_new,colors):
        indicesToKeep = targetK
        ax2.scatter(finalDf_new.loc[indicesToKeep.to_numpy().reshape(-1), 'PC1']
                   , finalDf_new.loc[indicesToKeep.to_numpy().reshape(-1), 'PC2']
                   , c = color
                   , s = 20) 

    plt.xticks(np.arange(-12, 14, 12)) 
    plt.yticks(np.arange(-6, 8, 6))      
    ax2.legend(['<7', '7-9','>9'], fontsize=18)    
    plt.show()
            
    fig2.savefig('lassoPCA_new.pdf')
            
    return outliner

def func3DPCA(tol, target, finalDf, PCs):
    # plot 3-D PCA figurre
    fig3D = plt.figure(figsize = (8,8))
    ax = fig3D.add_subplot(1,1,1, projection='3d')

    ax.set_xlim(-30,32)
    ax.set_ylim(-30,32)
    ax.set_zlim(-30,32)
    
    ax.set_xlabel('PC 1 ('+str(round(PCs[0]*100,2))+'%)', fontsize = 15)
    ax.set_ylabel('PC 2 ('+str(round(PCs[1]*100,2))+'%)', fontsize = 15)
    ax.set_zlabel('PC 3 ('+str(round(PCs[2]*100,2))+'%)', fontsize = 15)
    ax.set_title('Grouped by age', fontsize = 20)
 
    targetcol = [target < 7,np.logical_and(target >= 7, target <= 9), target > 9]
    colors = ['r', 'b', 'g']
    for targetK, color in zip(targetcol,colors):
        indicesToKeep = targetK
        ax.scatter(finalDf.loc[indicesToKeep.to_numpy().reshape(-1), 'PC1']
                   , finalDf.loc[indicesToKeep.to_numpy().reshape(-1), 'PC2']
                   , finalDf.loc[indicesToKeep.to_numpy().reshape(-1), 'PC3']
                   , c = color
                   , s = 20) 
        
    ax.legend(['<7', '7-9','>9'])    
    # Set rotation angle to 30 degrees
    ax.view_init(15,200)
    plt.show()
            
    fig3D.savefig('lassoPCA3D.pdf')
        
    # identifiy the outliners
    L = len(target)
    outliner = []
    for i in range(L):
        dis = finalDf['PC1'][i]**2 + finalDf['PC2'][i]**2 + finalDf['PC3'][i]**2
        if dis > tol**2:
            outliner = np.append(outliner, i)
            
    #PCA plot without outliners
    finalDf_new = finalDf.drop(outliner)
    finalDf_new.reset_index(drop = True, inplace = True)
    target_new = target.drop(outliner)
    target_new.reset_index(drop = True, inplace = True)
    fig3D2 = plt.figure(figsize = (8,8))
    ax2 = fig3D2.add_subplot(1,1,1, projection='3d')

    ax2.set_xlim(-30,32)
    ax2.set_ylim(-30,32)
    ax2.set_zlim(-30,32)
    
    ax2.set_xlabel('PC 1 ('+str(round(PCs[0]*100,2))+'%)', fontsize = 15)
    ax2.set_ylabel('PC 2 ('+str(round(PCs[1]*100,2))+'%)', fontsize = 15)
    ax2.set_zlabel('PC 3 ('+str(round(PCs[2]*100,2))+'%)', fontsize = 15)
    ax2.set_title('Grouped by age', fontsize = 20)
 
    targetcol_new = [target_new < 7,np.logical_and(target_new >= 7, target_new <= 9), target_new > 9]
    for targetK, color in zip(targetcol_new,colors):
        indicesToKeep = targetK
        ax2.scatter(finalDf_new.loc[indicesToKeep.to_numpy().reshape(-1), 'PC1']
                   , finalDf_new.loc[indicesToKeep.to_numpy().reshape(-1), 'PC2']
                   , finalDf_new.loc[indicesToKeep.to_numpy().reshape(-1), 'PC3']
                   , c = color
                   , s = 20) 
        
    ax2.legend(['<7', '7-9','>9'])    
    # Set rotation angle to 30 degrees
    ax2.view_init(15,200)
    plt.show()
            
    fig3D2.savefig('lassoPCA3D_new.pdf')
    
    return outliner

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

# %% perform Lasso regreession
lasso_tuned = Lasso(alpha=0.01, max_iter=10e5)
lasso_tuned.fit(X,y)

# save coefficients
CoffInd = lasso_tuned.coef_
LassID = zip(feature_name,CoffInd)
LassTab = pd.DataFrame(LassID, columns =['Indices','Coef'])
LassTab = LassTab.iloc[(-LassTab['Coef'].abs()).argsort()].reset_index()
LassTab.to_csv('Importances_Lasso.csv')

# get the index for the key features
LassoKInd = abs(lasso_tuned.coef_) > 0.001


# %% PCA
# Separating out the features
feature_name_PCA = list(compress(feature_name,LassoKInd))
PCA_data = Twin_data[feature_name_PCA]
xPCA = PCA_data.loc[:, feature_name_PCA].values

# Standardizing the features
xPCA = StandardScaler().fit_transform(xPCA)

# perform PCA 
pca = PCA()
principalComponents = pca.fit_transform(xPCA)
loadings = pca.components_
n = loadings.shape[0]
PCs = pca.explained_variance_ratio_
# Construct dataframe for the PCs & target 
column = []
for i in range(n):
    column.append("PC"+str(i+1))
    
principalDf = pd.DataFrame(data = principalComponents, 
                           columns=column)
target = Twin_data[target_name] 
finalDf = pd.concat([principalDf.iloc[:,0:3], target], axis = 1)

# tolerance distance for identify outliner 
tol = 15

# plot PCA plots and find outliners
if PCs[0]+PCs[1] > 0.9:
    outliners = func2DPCA(tol, target, finalDf, PCs)
else:
    outliners = func3DPCA(tol, target, finalDf, PCs)

# get rid of the outliners
Twin_data_new = Twin_data.drop(outliners)
#Twin_data_new = Twin_data_new.query("PWV <= 20")
Twin_data_new.reset_index(drop = True, inplace = True)

# save the data without outliners
Twin_data_new.to_pickle('Twin_data_no_outliner.pkl')
