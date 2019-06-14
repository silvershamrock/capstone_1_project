#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:20:05 2019

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
from scipy.stats import uniform, 

#%% Prepare dataset and parameters
df, _ = prep.prepare_data('OnlineNewsPopularity.csv')

#Get attribute names
attributes = df.columns.drop('logshares')

#Get target data and full attribute data
y = df['logshares']
X = df[attributes]


#%% Do CV on full attribute set

# specify your configurations as a dict
params = {
    'boosting': 'gbdt',
    'objective': 'regression_l1',
    'n_estimators': 100,
    'learning_rate': 0.1
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
#Print error
print("The MAE in the last boosting round is: ", mae_total)

#%% Do Grid search to optimize parameters

#Create lgb regressor model
model = lgb.LGBMRegressor(boosting_type = 'gbrt',
                          objective = 'regression_l1',
                          random_state = 42,
                          n_estimators = 100,
                          learning_rate = .01,
                          max_depth = -1
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
params_opt =  {
                    'learning_rate': [0.01, .1],
                    'n_estimators': [80, 100],
                    'num_leaves': [64],
                    #'boosting_type' : ['gbdt'],
                    #'objective' : ['regression_l1'],
                    #'random_state' : [501], # Updated from 'seed'
                    'colsample_bytree' : [0.65],
                    'subsample' : [0.65],
                    'reg_alpha' : [1.2],
                    'reg_lambda' : [1.2],
                    }


#Run grid search
gridsearch = GridSearchCV(estimator = model, 
                          param_grid = params_opt,
                          cv = 5,
                          scoring = 'neg_mean_absolute_error')
gridsearch.fit(X, y)

#Print best score
print("The best negative MAE for GridSearch is: {}".format(gridsearch.best_score_))
