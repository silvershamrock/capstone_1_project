#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:29:02 2019

@author: ryanmiller
"""
#import modules
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from kmodes.kprototypes import KPrototypes
import prepare_data as prep
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Get group data  (ignore df, loading from file)
_, groups = prep.prepare_data('OnlineNewsPopularity.csv')

#Load final data with dropped variables
with open('/Users/ryanmiller/data science/capstone_1_project/data/Xy_dropweak.pckl', 'rb') as xyfile:
    X_dropweak, y = pickle.load(xyfile)
        
#Get attribute names
attributes = X_dropweak.columns

#Load CV model with best hyperparameters
with open('/Users/ryanmiller/data science/capstone_1_project/data/CVresults_dropweak.pckl', 'rb') as f:
    model_dropweak = pickle.load(f)

#Find categorical variables
catvars = ['data_channel_is_lifestyle', 'data_channel_is_entertainment',
           'data_channel_is_bus', 'data_channel_is_socmed', 
           'data_channel_is_tech', 'data_channel_is_world',
           'weekday_is_tuesday', 'weekday_is_wednesday', 
           'weekday_is_thursday', 'weekday_is_friday',
           'weekday_is_saturday', 'weekday_is_sunday']

catinds = [attributes.get_loc(c) for c in catvars if c in attributes]

# Get data centroids using K Prototypes
cost_array = {}
myscaler = StandardScaler()
X_scaled = myscaler.fit_transform(X_dropweak)
for k in range(1,11):
    kproto = KPrototypes(n_clusters=k, init='Cao', n_init=10, verbose=2)
    clusters = kproto.fit_predict(X_scaled, categorical=catinds)
    cost_array[k] = kproto.cost_

#plot cost function to look for elbow 
plt.figure()
plt.plot(list(cost_array.keys()), list(cost_array.values()))
plt.xlabel('Number of centroids (k)')
plt.ylabel('Cost Function (distances to centroid)')
plt.title('Elbow Plot for K Prototype Clustering')
#Save figure
plt.savefig("../reports/Kprototype_elbow_plot.png", dpi=300, bbox_inches='tight')

#Rerun best k
best_k = 10
kproto = KPrototypes(n_clusters=best_k, init='Cao', n_init=10, verbose=2)
clusters = kproto.fit_predict(X_scaled, categorical=catinds)

#Get indices of numerical features
numinds = [attributes.get_loc(a) for a in attributes if a not in catvars and a in attributes]

#Calculate distances from each article to centroid
distances = np.empty([len(X_dropweak), best_k])
for i in range(best_k):
    #First, for numerical features
    num_dist = np.sum((X_scaled[:,numinds] - kproto.cluster_centroids_[0][i])**2, axis=1)
    #Second, for categorical features
    cat_dist = np.sum(X_scaled[:,catinds] != kproto.cluster_centroids_[1][i], axis=1)
    distances[:,i] = num_dist + cat_dist
    
#Grab articles closest to centroids
article_inds = np.argmin(distances, axis=0)
rep_articles = X_dropweak.iloc[article_inds, :].copy()
rep_articles.reset_index(drop=True, inplace=True)
rep_array = np.array(rep_articles)

##Restructure centroid data to original form
#centroids = np.empty([k, len(attributes)])
#centroids[:,numinds] = kproto.cluster_centroids_[0]
#centroids[:,catinds] = kproto.cluster_centroids_[1]

##Convert scaled centroids to raw data and find representative (close) articles
#centroids_raw = myscaler.inverse_transform(centroids)
#for count, centroid in enumerate(centroids_raw):
    
#For each article, predict shares and tweak to increase shares
#features to tweak: codes: 
#0=continuous number, tweak -10 to 10% of median
#1=integer, tweak -10 to 10% of median, rounded
#2=integer, tweak -+5 integers
#3=binary, try both 0 and 1
features_tweak = {'kw_avg_avg': 0, 'self_reference_min_shares': 1,
                  'kw_max_avg': 0, 'kw_min_avg': 0, 
                  'self_reference_max_shares': 1, 'weekday_is_saturday': 3,
                  'weekday_is_sunday': 3, 'num_imgs': 2,
                  'num_hrefs': 2, 'num_videos': 2}
#day of week inds
day_inds = [attributes.get_loc(d) for d in groups['day_cols'] if d in attributes]

article_df = pd.DataFrame()
#loop artiles to tweak
for ind, article in enumerate(rep_array): 
    article_orig = article.reshape(1,-1).copy()
    shares_orig = np.e**model_dropweak.predict(article_orig)
    article_df.loc[ind,'shares_orig'] = shares_orig
    #loop features to tweak
    for feature, code in features_tweak.items():
        #Reset tweaked article to original for each feature
        article_tweak = article_orig.copy()
        feature_ind = X_dropweak.columns.get_loc(feature)
        feature_orig = article_tweak[0,feature_ind]
        article_df.loc[ind, feature + "_orig"] = feature_orig
        #tweak feature based on code
        if code == 0:
            test_vals = [feature_orig + p*X_dropweak[feature].median() 
                         for p in np.arange(-.1, .11, .01)
                         if feature_orig + p*X_dropweak[feature].median() >= 0]
        elif code == 1:
            test_vals = [np.round(feature_orig + p*X_dropweak[feature].median())
                         for p in np.arange(-.1, .11, .01)
                         if np.round(feature_orig + p*X_dropweak[feature].median()) >= 0]
        elif code == 2:
            test_vals = [feature_orig + p for p in np.arange(-5, 6)
                         if feature_orig + p >= 0]
        elif code == 3:
            test_vals = [0, 1]
        
        #loop test values and save best one
        best_val = feature_orig
        best_shares = shares_orig
        for val in test_vals:
            #if day of week, change other days of week to 0
            if code == 3 and val==1:
                article_tweak[0, day_inds] = 0
            #set feature val equal to test val
            article_tweak[0,feature_ind] = val
            #predict shares, if larger save as best val
            tweak_shares = np.e**model_dropweak.predict(article_tweak)
            if tweak_shares >= shares_orig:
                best_val = val
                best_shares = tweak_shares
        
        #Add tweak best value and best shares to df
        article_df.loc[ind, feature + "_best"] = best_val
        article_df.loc[ind, feature + "_shares"] = best_shares

#Grab shares columns and subtract original shares to get increases
article_shares_df = article_df.filter(regex='_shares$', axis=1).copy()
article_shares_df = article_shares_df.sub(article_df.shares_orig, axis=0)
article_shares_df.rename(columns = lambda x: x[0:-7], inplace=True)

#Plot swarmplot of share increases for each variable
df_melted = article_shares_df.melt()
sns.swarmplot(x='variable', y='value', data=df_melted)
plt.title('Increase in shares for 10 representative articles')
plt.ylabel('share increase')
plt.xlabel('')
plt.xticks(rotation=60)
