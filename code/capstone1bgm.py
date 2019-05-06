#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:27:58 2019

@author: ryanmiller
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn import preprocessing
import shap
import matplotlib.pyplot as plt
import os.path
import prepare_data as prep

#%% Prepare dataset and parameters
df = prep.prepare_data('OnlineNewsPopularity.csv')

#Get attribute names
attributes = df.columns.drop('logshares')

#Get target data and full attribute data
y = df['logshares']
X = df[attributes]


#%% Do CV on full attribute set

# specify your configurations as a dict
params = {
    'boosting': 'gbrt',
    'objective': 'regression_l1',
    'n_estimators': 100
}

#Create datasets for lgbm
lgb_train = lgb.Dataset(X, y)
   
#CV
cv_results = lgb.cv(params,
                    lgb_train, 
                    nfold=5,
                    metrics='l1',
                    stratified=False)

#grab error of last boosting round
mae_total = cv_results['l1-mean'][-1]

#%% Loop attributes, dropping one attribute at a time
#maedict = {}
#for a in attributes:
#    #Print name of attribute
#    print('Currently dropping: {}'.format(a))
#    #Create training data by dropping attribute
#    X1 = X.drop(a, axis=1)
#    
#    #Separate training and test data
#    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=21)
#    
#    #Create datasets for lgbm
#    lgb_train = lgb.Dataset(X_train, y_train)
#    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
#
#    # train model
#    gbm = lgb.train(params,
#                    lgb_train,
#                    valid_sets=lgb_eval)
#    
#    # predict
#    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
#    # eval and add error to 
#    maedict[a] = mean_absolute_error(y_test, y_pred)
    
#%% Loop drop attributes and do CV
maedict = {}
for a in attributes:
    #Print name of attribute
    print('Currently dropping: {}'.format(a))
    #Create training data by dropping attribute
    X1 = X.drop(a, axis=1)
    
    #Create datasets for lgbm
    lgb_train = lgb.Dataset(X1, y)
   
    #CV
    cv_results = lgb.cv(params,
                        lgb_train, 
                        nfold=5,
                        metrics='l1',
                        stratified=False)
    
    #add last boosting round error to dict
    maedict[a] = [cv_results['l1-mean'][-1], 
                  cv_results['l1-stdv'][-1],
                  cv_results['l1-mean'][-1] - mae_total]


#%% Create dataframe with error data
df_mae = pd.DataFrame.from_dict(maedict, 
                            orient='index', 
                            columns=['mae_mean_without',
                                     'mae_std_without',
                                     'mae_diff_from_full'])

#%% Plot va
plotdata = df_mae.sort_values('mae_diff_from_full')
fig, ax = plt.subplots()
ax.bar(x=plotdata.index, height=plotdata.mae_diff_from_full)
ax.set_ylabel('Increase in MAE following removal')
ax.set_xlabel('Attribute removed')
#ax.set_xticklabels(labels=plotdata.index, rotation=60)


#%%Do PCA on variables whose inclusion decrease error by <.0005
#Grab data for weak variables
df_weak = df[df_mae[df_mae['mae_diff_from_full']<.0005].index]
#Scale data for PCA
df_weak_scaled = preprocessing.scale(df_weak)
#Do PCA
pca = PCA()
pca.fit(df_weak_scaled)
pca_data = pca.transform(df_weak_scaled)
#Generate scree plot
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
#Plot loading scores for first and 2nd variables
pca_df = pd.DataFrame(pca_data, columns=labels)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('PCA Graph for first two PC')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.show()
#Look at variables that contributed most to PC1
loading_scores = pd.Series(pca.components_[0], index=df_weak.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_vars = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_vars])

#%% Tune hyperparamters

#Create lgb regressor model
model = lgb.LGBMRegressor(boosting_type = 'gbrt',
                          objective = 'regression',
                          learning_rate = 0.01,
                          colsample_bytree = .7,
                          max_depth = 6,
                          min_child_samples=60,
                          num_leaves = 60,
                          #n_estimators = 1000,
                          random_state = 42,
                          )
   
#parameters to optimize
params_opt = {
        #'max_depth': range(4, 9, 2),
        'n_estimators': [2000],
        #'min_child_samples': range(20, 101, 10)
        #'num_leaves': range(20, 101, 20)
        #'colsample_bytree': np.arange(.1, 1.01, .1)
        
        }

#Run grid search
gridsearch = GridSearchCV(estimator = model, 
                          param_grid = params_opt,
                          cv = 5,
                          scoring = 'neg_mean_absolute_error')
gridsearch.fit(X, y)

#Print best params
print("The best params are: {}".format(gridsearch.best_params_))


#%% Plot shap values and importances
## Create object that can calculate shap values
#explainer = shap.TreeExplainer(gbm)
#
## calculate shap values. This is what we will plot.
## Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
#shap_values = explainer.shap_values(X_test)
#
## Make shap plot
#shap.summary_plot(shap_values, X_test)
#
#print('Plotting feature importances...')
#ax = lgb.plot_importance(gbm, max_num_features=20)
#plt.show()
#
#ax = lgb.plot_importance(gbm, max_num_features=20, importance_type='gain')
#plt.show()

#%% Testing ground to figure out differences in output

#Create lgb regressor model
model = lgb.LGBMRegressor(boosting_type = 'gbrt',
                          objective = 'regression_l1',
                          random_state = 42,
                          n_estimators = 100,
                          learning_rate = .01
                          )
   
#parameters to optimize
#params_opt = {
#        #'max_depth': range(4, 9, 2),
#        #'n_estimators': [100],
#        #'min_child_samples': range(20, 101, 10)
#        #'num_leaves': range(20, 101, 20)
#        #'colsample_bytree': np.arange(.1, 1.01, .1)
#        
#        }
param_opt =  {
                    'learning_rate': [0.0065, 0.007, 0.0075],
                    'n_estimators': [70, 80, 90, 100],
                    'num_leaves': [8, 16, 32],
                    'boosting_type' : ['gbdt'],
                    'objective' : ['regression_l1'],
                    'random_state' : [501], # Updated from 'seed'
                    'colsample_bytree' : [0.64, 0.65, 0.66],
                    'subsample' : [0.65, 0.7, 0.75, 0.8],
                    'reg_alpha' : [1, 1.2],
                    'reg_lambda' : [1, 1.2, 1.4],
                    }


#Run grid search
gridsearch = GridSearchCV(estimator = model, 
                          param_grid = params_opt,
                          cv = 5,
                          scoring = 'neg_mean_absolute_error')
gridsearch.fit(X, y)

#Print best score
print("The best score for GridSearch is: {}".format(gridsearch.best_score_))

##### Try lgb.cv
# specify your configurations as a dict
params = {
    'boosting': 'gbrt',
    'objective': 'regression_l1',
    'num_boost_round': 500,
    'random_state': 42,
    
    'max_depth': -1,
    'num_leaves': 31,
    'subsample': 1,
    'colsample_bytree': 1,
    'learning_rate': .01,
    'min_child_weight': .001
}

#Create datasets for lgbm
lgb_train = lgb.Dataset(X, y)
   
#CV
cv_results = lgb.cv(params,
                    lgb_train, 
                    nfold=5,
                    metrics='l1',
                    stratified=False)

#grab error of last boosting round
print("The best score for lgb.cv is: {}".format(cv_results['l1-mean'][-1]))
