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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn import preprocessing
import shap
import matplotlib.pyplot as plt
import os
import os.path
import prepare_data as prep
import luigi
from scipy.stats import uniform, randint
from time import time
import pickle

#if os.path.exists("/Users/ryanmiller/data science/capstone_1_project/data/"
#          "RandomizedSearch_test.txt"):
#    os.remove("/Users/ryanmiller/data science/capstone_1_project/data/"
#              "RandomizedSearch_test.txt")

class RunGridSearchCV(luigi.Task):
    def output(self):
        return luigi.LocalTarget("/Users/ryanmiller/data science/capstone_1_project/data/GridSearch_final.txt")
    
    def run(self):
        df, _ = prep.prepare_data('OnlineNewsPopularity.csv')

        #Get attribute names
        attributes = df.columns.drop('logshares')

        #Get target data and full attribute data
        y = df['logshares']
        X = df[attributes]
    
    
        #Create lgb regressor model
        model = lgb.LGBMRegressor(boosting_type = 'gbrt',
                              objective = 'regression_l1',
                              random_state = 42,
                              n_estimators = 100,
                              learning_rate = .1,
                              num_leaves = 32,
                              max_depth = -1
                              )
       
        #parameters to optimize
        params_opt =  {
                        'learning_rate': [0.05, 0.1],
                        'n_estimators': [80, 100],
                        #'num_leaves': [8, 16, 32],
                        #'boosting_type' : ['gbdt'],
                        #'objective' : ['regression_l1'],
                        #'random_state' : [501], # Updated from 'seed'
                        #'colsample_bytree' : [0.64, 0.65, 0.66],
                        #'subsample' : [0.65, 0.7, 0.75, 0.8],
                        #'reg_alpha' : [1, 1.2],
                        #'reg_lambda' : [1, 1.2, 1.4],
                        }
    
    
        #Run grid search
        gridsearch = GridSearchCV(estimator = model, 
                                  param_grid = params_opt,
                                  cv = 5,
                                  scoring = 'neg_mean_absolute_error')
        gridsearch.fit(X, y)
        
        with self.output().open('w') as out_file:
            out_file.write("This is a test.\n")
            out_file.write("The best score is: {}".format(gridsearch.best_score_))

class RunRandomizedSearchCV(luigi.Task):
    def output(self):
        return luigi.LocalTarget("/Users/ryanmiller/data science/capstone_1_project/data/RandomizedSearch_new2.txt")
    
    def run(self):
        
        # Utility function to report best scores
        def report(results, n_top=10):
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                    out_file.write("Model with rank: {0}\n".format(i))
                    out_file.write("Mean validation score: {0:.3f} (std: {1:.3f})\n".format(
                                  results['mean_test_score'][candidate],
                                  results['std_test_score'][candidate]))
                    out_file.write("Parameters: {0}\n\n".format(results['params'][candidate]))
        
        
        df, _ = prep.prepare_data('OnlineNewsPopularity.csv')

        #Get attribute names
        attributes = df.columns.drop('logshares')

        #Shuffle rows, Get target data and full attribute data
        df_shuffled = df.sample(frac=1, random_state=42)
        df_shuffled.reset_index(drop=True, inplace=True)
        
        y = df_shuffled['logshares']
        X = df_shuffled[attributes]
        
        #Separate validation data
        X_leftover, X_test, y_leftover, y_test = train_test_split(X, y, test_size=0.1, random_state=200)
    
        #Create lgb regressor model
        model = lgb.LGBMRegressor(boosting_type = 'gbrt',
                              objective = 'regression_l1',
                              random_state = 42,
                              #n_estimators = 100,
                              #learning_rate = .1,
                              #num_leaves = 32,
                              #max_depth = -1
                              )
       
        #parameters to optimize
#        params_opt =  {
#                        'learning_rate': uniform(loc=0.001, scale=.199),
#                        'n_estimators': randint(50, 500),
#                        'num_leaves': randint(6, 40),
#                        'max_depth': [4, 8, 16, 24, None],
#                        'colsample_bytree' : uniform(loc=0.3, scale=.7),
#                        'subsample' : uniform(loc=.3, scale=.7),
#                        'reg_alpha' : uniform(loc=0, scale=1.4),
#                        'reg_lambda' : uniform(loc=0, scale=1.4),
#                        'random_state' : [33]
#                        }
        params_opt =  {
                        'learning_rate': uniform(loc=0.005, scale=.015),
                        'n_estimators': randint(2000, 3000),
                        'num_leaves': randint(30, 40),
                        'max_depth': [8, 16, None],
                        'colsample_bytree' : uniform(loc=0.2, scale=.2),
                        'subsample' : uniform(loc=0.3, scale=.4),
                        'reg_alpha' : uniform(loc=.1, scale=1.2),
                        'reg_lambda' : uniform(loc=.1, scale=1.2),
                        'random_state' : [33]
                        }
    
        # run randomized search
        n_iter_search = 200
        random_search = RandomizedSearchCV(estimator = model, 
                                           param_distributions=params_opt,
                                           n_iter=n_iter_search, 
                                           cv=5,
                                           random_state = 100,
                                           scoring = 'neg_mean_absolute_error')
        start = time()
        random_search.fit(X_leftover, y_leftover)
        
        #Save output to file for use in prediction later
        f = open('/Users/ryanmiller/data science/capstone_1_project/data/CVresults_final.pckl', 'wb')
        pickle.dump(random_search, f)
        f.close()
        
        with self.output().open('w') as out_file:
            out_file.write("RandomizedSearchCV took %.2f seconds for %d candidates"
                           " parameter settings.\n\n" % ((time() - start), n_iter_search))
            report(random_search.cv_results_)

class RunRandomizedSearchCV_dropweak(luigi.Task):
    def output(self):
        return luigi.LocalTarget("/Users/ryanmiller/data science/capstone_1_project/data/RandomizedSearch_dropweak.txt")
    
    def run(self):
        
        # Utility function to report best scores
        def report(results, n_top=10):
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                    out_file.write("Model with rank: {0}\n".format(i))
                    out_file.write("Mean validation score: {0:.3f} (std: {1:.3f})\n".format(
                                  results['mean_test_score'][candidate],
                                  results['std_test_score'][candidate]))
                    out_file.write("Parameters: {0}\n\n".format(results['params'][candidate]))
        
        
        with open('/Users/ryanmiller/data science/capstone_1_project/data/Xy_dropweak.pckl', 'rb') as xyfile:
            X_dropweak, y = pickle.load(xyfile)
        
        #Separate validation data
        X_leftover, X_test, y_leftover, y_test = train_test_split(X_dropweak, y, test_size=0.1, random_state=200)
    
        #Create lgb regressor model
        model = lgb.LGBMRegressor(boosting_type = 'gbrt',
                              objective = 'regression_l1',
                              random_state = 42,
                              #n_estimators = 100,
                              #learning_rate = .1,
                              #num_leaves = 32,
                              #max_depth = -1
                              )
       
        #parameters to optimize
#        params_opt =  {
#                        'learning_rate': uniform(loc=0.001, scale=.199),
#                        'n_estimators': randint(50, 500),
#                        'num_leaves': randint(6, 40),
#                        'max_depth': [4, 8, 16, 24, None],
#                        'colsample_bytree' : uniform(loc=0.3, scale=.7),
#                        'subsample' : uniform(loc=.3, scale=.7),
#                        'reg_alpha' : uniform(loc=0, scale=1.4),
#                        'reg_lambda' : uniform(loc=0, scale=1.4),
#                        'random_state' : [33]
#                        }
        params_opt =  {
                        'learning_rate': uniform(loc=0.005, scale=.015),
                        'n_estimators': randint(2000, 3000),
                        'num_leaves': randint(30, 40),
                        'max_depth': [8, 16, None],
                        'colsample_bytree' : uniform(loc=0.2, scale=.2),
                        'subsample' : uniform(loc=0.3, scale=.4),
                        'reg_alpha' : uniform(loc=.1, scale=1.2),
                        'reg_lambda' : uniform(loc=.1, scale=1.2),
                        'random_state' : [33]
                        }
    
        # run randomized search
        n_iter_search = 200
        random_search = RandomizedSearchCV(estimator = model, 
                                           param_distributions=params_opt,
                                           n_iter=n_iter_search, 
                                           cv=5,
                                           random_state = 100,
                                           scoring = 'neg_mean_absolute_error')
        start = time()
        random_search.fit(X_leftover, y_leftover)
        
        #Save output to file for use in prediction later
        f = open('/Users/ryanmiller/data science/capstone_1_project/data/CVresults_dropweak.pckl', 'wb')
        pickle.dump(random_search, f)
        f.close()
        
        with self.output().open('w') as out_file:
            out_file.write("RandomizedSearchCV took %.2f seconds for %d candidates"
                           " parameter settings.\n\n" % ((time() - start), n_iter_search))
            report(random_search.cv_results_)

if __name__ == "__main__":
    luigi.build([RunRandomizedSearchCV()], local_scheduler=True)