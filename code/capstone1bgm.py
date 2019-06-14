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
df, groups = prep.prepare_data('OnlineNewsPopularity.csv')

#Get attribute names
attributes = df.columns.drop('logshares')

#Shuffle rows data before doing decision trees
df_shuffled = df.sample(frac=1, random_state=42)
df_shuffled.reset_index(drop=True, inplace=True)
        
y = df_shuffled['logshares']
X = df_shuffled[attributes]

# specify your configurations as a dict
params = {
    'boosting': 'gbdt',
    'objective': 'regression_l1',
    'n_estimators': 359,
    'colsample_bytree': .34,
    'max_depth': 8,
    'min_child_samples': 100,
    'learning_rate': .043,
    'num_leaves': 38,
    'random_state': 500,
    'reg_alpha': .22,
    'reg_lambda': 1.18,
    'subsample': .66
}
params_alt = {
    'boosting': 'gbdt',
    'objective': 'regression_l1',
    'n_estimators': 2500,
    'learning_rate': .007, #.007 best so far
    'random_state': 500,
    'colsample_bytree': .34,
    'max_depth': 8,
    'min_child_samples': 100, #100 best so far
    'num_leaves': 38,
    'subsample': .9, #.66 best so far
    'subsample_freq': 1,
    'reg_alpha': .3,
    'reg_lambda': 1.18
}


#%% Fit regular GBM model without CV

#Separate validation data
X_leftover, X_test, y_leftover, y_test = train_test_split(X, y, test_size=0.1, random_state=200)

#Separate test data
X_train, X_valid, y_train, y_valid = train_test_split(X_leftover, y_leftover, test_size=0.1, random_state=300)

#Create datasets for lgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# train model
gbm = lgb.train(params_alt,
                lgb_train,
                valid_sets=lgb_eval)
                #early_stopping_rounds=20)

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval and add error to 
mae_regfull = mean_absolute_error(y_test, y_pred)
print("MAE for regular train and predict: ", mae_regfull)

#mae_20stopping = mae_regfull

#%% Do CV on full attribute set

#Create datasets for lgbm
lgb_train = lgb.Dataset(X, y)
   
#CV
cv_results = lgb.cv(params,
                    lgb_train, 
                    nfold=5,
                    metrics = 'l1',
                    stratified=False)

#grab error of last boosting round
mae_total = cv_results['l1-mean'][-1]
print("MAE from regular CV: ", mae_total)

#%% Use scikit-learn API without CV to test if same result as basic train

model = lgb.LGBMRegressor(boosting_type = 'gbdt',
                          objective = 'regression_l1',
                          colsample_bytree = .34,
                          max_depth = 8,
                          min_child_samples = 100,
                          learning_rate = .043,
                          num_leaves = 38,
                          n_estimators = 359,
                          random_state = 500,
                          reg_alpha = .22,
                          reg_lambda = 1.18,
                          subsample = .66
                          )

model.fit(X_train, y_train)
y_pred = model.predict(X_test, num_iteration=gbm.best_iteration)
mae_regfull_scikit = mean_absolute_error(y_test, y_pred)
print("MAE from scikit without CV: ", mae_regfull_scikit)

#%% Loop attributes, dropping one attribute at a time
maedict = {}

#Separate validation data
X_leftover, X_test, y_leftover, y_test = train_test_split(X, y, test_size=0.1, random_state=200)

#Separate test data
X_train, X_valid, y_train, y_valid = train_test_split(X_leftover, y_leftover, test_size=0.1, random_state=300)

for a in attributes:
    #Print name of attribute
    print('Currently dropping: {}'.format(a))
    #Create training data by dropping attribute
    X1_train = X_train.drop(a, axis=1)
    X1_valid = X_valid.drop(a, axis=1)
    X1_test = X_test.drop(a, axis=1)
    
    #Create datasets for lgbm
    lgb_train = lgb.Dataset(X1_train, y_train)
    lgb_eval = lgb.Dataset(X1_valid, y_valid, reference=lgb_train)

    # train model
    gbm = lgb.train(params_alt,
                    lgb_train,
                    valid_sets=lgb_eval)
    
    # predict
    y_pred = gbm.predict(X1_test, num_iteration=gbm.best_iteration)
    # eval and add error to 
    maedict[a] = [mean_absolute_error(y_test, y_pred),
                  mean_absolute_error(y_test, y_pred) - mae_regfull]
    
#%% Loop drop attributes and do CV
#maedict = {}
#
#for a in attributes:
#    #Print name of attribute
#    print('Currently dropping: {}'.format(a))
#    #Create training data by dropping attribute
#    X1 = X.drop(a, axis=1)
#    
#    #Create datasets for lgbm
#    lgb_train = lgb.Dataset(X1, y)
#   
#    #CV
#    cv_results = lgb.cv(params,
#                        lgb_train, 
#                        nfold=5,
#                        metrics='l1',
#                        stratified=False)
#    
#    #add last boosting round error to dict
#    maedict[a] = [cv_results['l1-mean'][-1], 
#                  cv_results['l1-stdv'][-1],
#                  cv_results['l1-mean'][-1] - mae_total]
#

#%% Create dataframe with error data
df_mae = pd.DataFrame.from_dict(maedict, 
                            orient='index', 
                            columns=['mae_without',
                                     'mae_diff_from_full'])

#%% Plot MAE value decreases for each attribute
plotdata = df_mae.sort_values('mae_diff_from_full')
fig, ax = plt.subplots()
ax.bar(x=plotdata.index, height=plotdata.mae_diff_from_full)
ax.set_ylabel('Increase in MAE following removal')
ax.set_xlabel('Attribute removed')
#ax.set_xticklabels(labels=plotdata.index, rotation=60)


#%%Do PCA on variables whose inclusion decrease error by <= 0
#Grab data for weak variables
pcvars = df_mae[df_mae['mae_diff_from_full']<=0.01].index
df_weak = X[pcvars]
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

#%%Do PCA on polar variables since they are highly correlated
#Grab data for weak variables
df_polar = X[groups['polar_cols']]
#Scale data for PCA
df_polar_scaled = preprocessing.scale(df_polar)
#Do PCA
pca = PCA()
pca.fit(df_polar_scaled)
pca_data = pca.transform(df_polar_scaled)
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
loading_scores = pd.Series(pca.components_[0], index=df_polar.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_vars = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_vars])

#%% Add back in one PC at a time to data without polar and compare to previous accuracy

#remove polar features
X_prev = X.drop(pcvars, axis=1)

#Separate validation data
X_leftoverprev, X_testprev, y_leftover, y_test = train_test_split(X_prev, y, test_size=0.1, random_state=200)

#Separate test data
X_trainprev, X_validprev, y_train, y_valid = train_test_split(X_leftoverprev, y_leftover, test_size=0.1, random_state=300)


#X_trainprev = X_train.drop(pcvars, axis=1)
#X_validprev = X_valid.drop(pcvars, axis=1)
#X_testprev = X_test.drop(pcvars, axis=1)
diff_dict = {}
mae_when_added = {}

# Do CV on starting attribute set
#Create datasets for lgbm
#lgb_train = lgb.Dataset(X_prev, y)   
#Create datasets for lgbm
lgb_train = lgb.Dataset(X_trainprev, y_train)
lgb_eval = lgb.Dataset(X_validprev, y_valid, reference=lgb_train)

# train model
gbm = lgb.train(params_alt,
                lgb_train,
                valid_sets=lgb_eval)
                #early_stopping_rounds=50)

# predict
y_pred = gbm.predict(X_testprev, num_iteration=gbm.best_iteration)
# eval and add error
mae_prev = mean_absolute_error(y_test, y_pred)
mae_when_added['base'] = mae_prev

#Loop through PCs adding one at a time and compare to previous results
for pc in pca_df.columns:   
    #Add next PC and Do CV
    X_prev_addone = X_prev.join(pca_df[pc])
    #Separate validation data
    X_leftover_addone, X_test_addone, y_leftover, y_test = train_test_split(X_prev_addone, y, test_size=0.1, random_state=200)

    #Separate test data
    X_train_addone, X_valid_addone, y_train, y_valid = train_test_split(X_leftover_addone, y_leftover, test_size=0.1, random_state=300)

    #X_addone = X_prev.join(df_polar[var])
    #Create datasets for lgbm
    lgb_train = lgb.Dataset(X_train_addone, y_train)  
    lgb_eval = lgb.Dataset(X_valid_addone, y_valid, reference=lgb_train)
    
    # train model
    gbm = lgb.train(params_alt,
                    lgb_train,
                    valid_sets=lgb_eval)
                    #early_stopping_rounds=50)
    # predict
    y_pred = gbm.predict(X_test_addone, num_iteration=gbm.best_iteration)
    # eval and add error
    mae_addone = mean_absolute_error(y_test, y_pred)  
              
    diff_dict[pc] = mae_prev - mae_addone
    mae_when_added[pc] = mae_addone
    
    print("Now adding {}....".format(pc))
    print("MAE from previous CV: ", mae_prev)
    print("MAE from added one CV: ", mae_addone)
    print("Diff in MAE: ", mae_prev - mae_addone)
    
    #Set X_prev to X_addone for doing next loop
    X_prev = X_prev_addone
#    X_validprev = X_valid_addone
#    X_testprev = X_test_addone
    mae_prev = mae_addone
    
    


#%% Tune hyperparamters

#Create lgb regressor model
model = lgb.LGBMRegressor(boosting_type = 'gbdt',
                          objective = 'regression_l1',
                          colsample_bytree = .3,
                          max_depth = 8,
                          min_child_samples = 100,
                          learning_rate = .1,
                          num_leaves = 30,
                          n_estimators = 100,
                          random_state = 300
                          )
   
#parameters to optimize
params_opt = {
        #'max_depth': [4, 6, 8, 10],
        #'n_estimators': [100, 120, 140],
        #'learning_rate': [.045, .05, .055],
        #'min_child_samples': [15, 20, 25]
        #'num_leaves': [35, 40, 45]
        #'colsample_bytree': [.35, .4, .45]
        
        }

#Run grid search
gridsearch = GridSearchCV(estimator = model, 
                          param_grid = params_opt,
                          cv = 5,
                          scoring = 'neg_mean_absolute_error')

#Fit the model
gridsearch.fit(X_train, y_train)

y_pred = gridsearch.predict(X_test)
mae_fromgridsearch = mean_absolute_error(y_test, y_pred)
print("MAE on test set from GridSearchCV: ", mae_fromgridsearch)

#Print best params
print("The best params are: {}".format(gridsearch.best_params_))
print("The best score is: {}".format(gridsearch.best_score_))


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
params_opt =  {
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
