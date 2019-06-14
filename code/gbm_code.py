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
import os
import prepare_data as prep
import pickle

#%% Prepare dataset and parameters
df, groups = prep.prepare_data('OnlineNewsPopularity.csv')

#Get attribute names
attributes = df.columns.drop('logshares')

#Shuffle rows data before doing decision trees
df_shuffled = df.sample(frac=1, random_state=42)
df_shuffled.reset_index(drop=True, inplace=True)
        
y = df_shuffled['logshares']
X = df_shuffled[attributes]


#%% Fit regular GBM model to training data 

#Separate out test data
X_leftover, X_test, y_leftover, y_test = train_test_split(X, y, test_size=0.1, random_state=200)

#Separate remainder into training and validation data for optimization
X_train, X_valid, y_train, y_valid = train_test_split(X_leftover, y_leftover, test_size=0.2, random_state=300)

#Load CV results
with open('/Users/ryanmiller/data science/capstone_1_project/data/CVresults_final.pckl' ,'rb') as datafile:
    cv_results = pickle.load(datafile)

#Retrieve best estimator from results
model = cv_results.best_estimator_

#Train model on training data
model.fit(X_train, y_train, eval_set=(X_valid, y_valid)) 

# predict valid data
y_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
# evaluate accuracy 
mae_regfull = mean_absolute_error(y_valid, y_pred)
print("MAE for validation data: ", mae_regfull)

#%% Loop attributes, dropping one attribute at a time
maedict = {}
#Duplicate model so original isn't overwritten
model1 = model
for a in attributes:
    #Print name of attribute
    print('Currently dropping: {}'.format(a))
    #Create training data by dropping attribute
    X1_train = X_train.drop(a, axis=1)
    X1_valid = X_valid.drop(a, axis=1)
    X1_test = X_test.drop(a, axis=1)
    
    # train model
    model1.fit(X1_train, y_train, eval_set=(X1_valid, y_valid))
    
    # predict on validation set
    y_pred = model1.predict(X1_valid, num_iteration=model1.best_iteration_)
    # eval and add error to 
    maedict[a] = [mean_absolute_error(y_valid, y_pred),
                  mean_absolute_error(y_valid, y_pred) - mae_regfull]


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
pcvars = df_mae[df_mae['mae_diff_from_full']<=0].index
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
#Save PCA data to df
pca_df = pd.DataFrame(pca_data, columns=labels)
#Plot loading scores for first and 2nd variables
#plt.scatter(pca_df.PC1, pca_df.PC2)
#plt.title('PCA Graph for first two PC')
#plt.xlabel('PC1 - {0}%'.format(per_var[0]))
#plt.ylabel('PC2 - {0}%'.format(per_var[1]))
#plt.show()
#Look at variables that contributed most to PC1
loading_scores = pd.Series(pca.components_[0], index=df_weak.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_vars = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_vars])

##%%Do PCA on polar variables since they are highly correlated
##Grab data for weak variables
#df_polar = X[groups['polar_cols']]
##Scale data for PCA
#df_polar_scaled = preprocessing.scale(df_polar)
##Do PCA
#pca = PCA()
#pca.fit(df_polar_scaled)
#pca_data = pca.transform(df_polar_scaled)
##Generate scree plot
#per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
#labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
#plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
#plt.ylabel('Percentage of Explained Variance')
#plt.xlabel('Principal Component')
#plt.title('Scree Plot')
#plt.show()
##Plot loading scores for first and 2nd variables
#pca_df = pd.DataFrame(pca_data, columns=labels)
#plt.scatter(pca_df.PC1, pca_df.PC2)
#plt.title('PCA Graph for first two PC')
#plt.xlabel('PC1 - {0}%'.format(per_var[0]))
#plt.ylabel('PC2 - {0}%'.format(per_var[1]))
#plt.show()
##Look at variables that contributed most to PC1
#loading_scores = pd.Series(pca.components_[0], index=df_polar.columns)
#sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
#top_10_vars = sorted_loading_scores[0:10].index.values
#print(loading_scores[top_10_vars])

#%% Add back in one PC at a time to data without polar and compare to previous accuracy

#remove weak features
X_prev = X.drop(pcvars, axis=1)

#Separate out test data
X_leftoverprev, X_testprev, y_leftover, y_test = train_test_split(X_prev, y, test_size=0.1, random_state=200)

#Separate validation data from train data
X_trainprev, X_validprev, y_train, y_valid = train_test_split(X_leftoverprev, y_leftover, test_size=0.2, random_state=300)

#initialize dictionaries to save output
diff_dict = {}
mae_when_added = {}

#duplicate original model
model_noweak = model

# train model
model_noweak.fit(X_trainprev, y_train, eval_set=(X_validprev, y_valid))

#predict on validation data
y_pred = model_noweak.predict(X_validprev, num_iteration=model_noweak.best_iteration_)

#evaluate error 
mae_prev = mean_absolute_error(y_valid, y_pred)
mae_when_added['base'] = mae_prev

#Loop through PCs adding one at a time and compare to previous results
for pc in pca_df.columns:   
    #Add next PC and Do CV
    X_prev_addone = X_prev.join(pca_df[pc])
    #Separate validation data
    X_leftover_addone, X_test_addone, y_leftover, y_test = train_test_split(X_prev_addone, y, test_size=0.1, random_state=200)

    #Separate test data
    X_train_addone, X_valid_addone, y_train, y_valid = train_test_split(X_leftover_addone, y_leftover, test_size=0.2, random_state=300)
    
    #Create datasets for lgbm
    
    lgb_train = lgb.Dataset(X_train_addone, y_train)  
    lgb_eval = lgb.Dataset(X_valid_addone, y_valid, reference=lgb_train)
    
    # train model
    model_noweak.fit(X_train_addone, y_train, eval_set=(X_valid_addone, y_valid))
    
    # predict on validation data
    y_pred = model_noweak.predict(X_valid_addone, num_iteration=model_noweak.best_iteration_)
    # evaluate
    mae_addone = mean_absolute_error(y_valid, y_pred)  
              
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

#%% Save X and y with dropped weak features for use in luigi CV optimization code
X_dropweak = X.drop(pcvars, axis=1)
with open('/Users/ryanmiller/data science/capstone_1_project/data/Xy_dropweak.pckl', 'wb') as xyfile:
    pickle.dump([X_dropweak, y], xyfile)


#%% Do final prediction of test data
#Load final model with dropped weak features
with open('/Users/ryanmiller/data science/capstone_1_project/data/CVresults_dropweak.pckl', 'rb') as modelfile:
    cv_results = pickle.load(modelfile)

model_dropweak = cv_results.best_estimator_

#Separate out test data
X_leftover_dropweak, X_test_dropweak, y_leftover, y_test = train_test_split(X_dropweak, y, test_size=0.1, random_state=200)


y_pred = model_dropweak.predict(X_test_dropweak, num_iteration=model_dropweak.best_iteration_)

mae_test = mean_absolute_error(y_test, y_pred)

print("MAE for final test data: ", mae_test)

#%% Plot shap values and importances
# Create object that can calculate shap values
explainer = shap.TreeExplainer(model_dropweak)

#calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(X_test_dropweak)

## Make shap plot
shap.summary_plot(shap_values, X_test_dropweak, show=False)
#Save figure
plt.savefig("../reports/shap_plot.png", dpi=300, bbox_inches='tight')
#Plot feature importances
print('Plotting feature importances...')
#ax = lgb.plot_importance(model_dropweak, max_num_features=20)
#plt.show()

ax = lgb.plot_importance(model_dropweak, max_num_features=20, importance_type='gain')
plt.show()


#%% Tune hyperparamters

##Create lgb regressor model
#model = lgb.LGBMRegressor(boosting_type = 'gbdt',
#                          objective = 'regression_l1',
#                          colsample_bytree = .3,
#                          max_depth = 8,
#                          min_child_samples = 100,
#                          learning_rate = .1,
#                          num_leaves = 30,
#                          n_estimators = 100,
#                          random_state = 300
#                          )
#   
##parameters to optimize
#params_opt = {
#        #'max_depth': [4, 6, 8, 10],
#        #'n_estimators': [100, 120, 140],
#        #'learning_rate': [.045, .05, .055],
#        #'min_child_samples': [15, 20, 25]
#        #'num_leaves': [35, 40, 45]
#        #'colsample_bytree': [.35, .4, .45]
#        
#        }
#
##Run grid search
#gridsearch = GridSearchCV(estimator = model, 
#                          param_grid = params_opt,
#                          cv = 5,
#                          scoring = 'neg_mean_absolute_error')
#
##Fit the model
#gridsearch.fit(X_train, y_train)
#
#y_pred = gridsearch.predict(X_test)
#mae_fromgridsearch = mean_absolute_error(y_test, y_pred)
#print("MAE on test set from GridSearchCV: ", mae_fromgridsearch)
#
##Print best params
#print("The best params are: {}".format(gridsearch.best_params_))
#print("The best score is: {}".format(gridsearch.best_score_))




