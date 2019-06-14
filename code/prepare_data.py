#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:33:14 2019

@author: ryanmiller
"""
import os.path
import pandas as pd
import numpy as np
import re

def prepare_data(filename):
    
    groups = {}
    
    groups['word_cols'] = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
                           'n_non_stop_unique_tokens', 'average_token_length']
    groups['media_cols'] = ['num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos']
    groups['cat_cols'] = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 
                          'data_channel_is_bus', 'data_channel_is_socmed', 
                          'data_channel_is_tech', 'data_channel_is_world']
    groups['keyword_cols'] = ['num_keywords', 'kw_min_min', 'kw_max_min', 
                              'kw_avg_min', 'kw_min_max', 'kw_max_max', 
                              'kw_avg_max', 'kw_min_avg', 'kw_max_avg',
                              'kw_avg_avg']
    groups['selfref_cols'] = ['self_reference_min_shares', 
                              'self_reference_max_shares',
                              'self_reference_avg_sharess']
    groups['day_cols'] = ['weekday_is_tuesday', 'weekday_is_wednesday', 
                          'weekday_is_thursday', 'weekday_is_friday',
                          'weekday_is_saturday', 'weekday_is_sunday']
    groups['lda_cols'] = ['LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04']
    groups['subj_cols'] = ['global_subjectivity', 'title_subjectivity',
                           'abs_title_subjectivity']
    groups['polar_cols'] = ['global_sentiment_polarity', 'global_rate_positive_words',
                            'global_rate_negative_words', 'rate_positive_words',
                            'rate_negative_words', 'avg_positive_polarity', 
                            'min_positive_polarity', 'max_positive_polarity',
                            'avg_negative_polarity', 'min_negative_polarity',
                            'max_negative_polarity', 'title_sentiment_polarity', 
                            'abs_title_sentiment_polarity'] 
    # Get path
    #dirpath = os.path.join(os.path.dirname(os.getcwd()), 'data')
    dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    filepath = os.path.join(dirpath, 'data', filename)
    # Prepare dataset and parameters
    df = pd.read_csv(filepath)

    #Strip white space from column names and add log shares
    df.columns = df.columns.str.strip()
    df['logshares'] = np.log(df['shares'])

    #Parse dates in URL and add month as feature
    df['month'] = df['url'].apply(lambda x: int(re.findall('20\d{2}/\d{2}/\d{2}', x)[0].split('/')[1]))    
    df['month_sin'] = np.sin((df.month-1)*(2.*np.pi/12))
    df['month_cos'] = np.cos((df.month-1)*(2.*np.pi/12))
    
    #Drop url, shares, weekday_is_monday, is_weekend
    df = df.drop(['shares', 'url', 'weekday_is_monday', 'is_weekend', 'month'], axis=1)
    
    #Drop n_non_stop_words due to weird scaling
    df = df.drop(['n_non_stop_words'], axis=1)
    
    #Drop rows with impossible values 
    df = df[df['average_token_length']>=1]
    df = df[df['n_unique_tokens']<=1]
    
    return df, groups