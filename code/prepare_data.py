#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:33:14 2019

@author: ryanmiller
"""
import os.path
import pandas as pd
import numpy as np

def prepare_data(filename):
    
    # Get path
    dirpath = os.path.join(os.path.dirname(os.getcwd()), 'data')
    filepath = os.path.join(dirpath, filename)
    # Prepare dataset and parameters
    df = pd.read_csv(filepath)

    #### Strip white space from column names and add log shares
    df.columns = df.columns.str.strip()
    df['logshares'] = np.log(df['shares'])
         
    #Drop url, shares, weekday_is_monday, is_weekend
    df = df.drop(['shares', 'url', 'weekday_is_monday', 'is_weekend'], axis=1)
    
    return df