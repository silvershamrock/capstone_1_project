# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import graphfuncs as gf
import os.path
import prepare_data as prep

df, groups = prep.prepare_data('OnlineNewsPopularity.csv')

word_cols = ['n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
             'n_non_stop_unique_tokens', 'average_token_length']
media_cols = ['num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos']
cat_cols = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 
            'data_channel_is_bus', 'data_channel_is_socmed', 
            'data_channel_is_tech', 'data_channel_is_world']
keyword_cols = ['num_keywords', 'kw_min_min', 'kw_max_min', 
                'kw_avg_min', 'kw_min_max', 'kw_max_max', 
                'kw_avg_max', 'kw_min_avg', 'kw_max_avg',
                'kw_avg_avg']
selfref_cols = ['self_reference_min_shares', 
                'self_reference_max_shares',
                'self_reference_avg_sharess']
day_cols = ['weekday_is_tuesday', 'weekday_is_wednesday', 
            'weekday_is_thursday', 'weekday_is_friday',
            'weekday_is_saturday', 'weekday_is_sunday']
lda_cols = ['LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04']
subj_cols = ['global_subjectivity', 'title_subjectivity',
             'abs_title_subjectivity']
polar_cols = ['global_sentiment_polarity', 'global_rate_positive_words',
              'global_rate_negative_words', 'rate_positive_words',
              'rate_negative_words', 'avg_positive_polarity', 
              'min_positive_polarity', 'max_positive_polarity',
              'avg_negative_polarity', 'min_negative_polarity',
              'max_negative_polarity', 'title_sentiment_polarity', 
              'abs_title_sentiment_polarity'] 

                
##### Plot timedelta column in histogram
#plt.figure()
#g = sns.distplot(df['timedelta'], bins=30, kde=False)
#plt.show()
#gf.get_dist_graph(df, 'timedelta')

##### Make scatter plot of timedelta and log shares
plt.figure()
g = sns.regplot(x=df.loc[df['timedelta']<10, 'timedelta'], y=df.loc[df['timedelta']<10, 'logshares'])
plt.show()

##### Make scatter plot of timedelta and log shares
plt.figure()
g = sns.regplot(x=df['timedelta'], y=df['logshares'],
                scatter_kws={"alpha":0.2, "s":5})
plt.xlabel("Time since publication")
plt.show()


#####Calculate correlation of word_cols with log shares correlation matrix
corrcols = [*groups['word_cols'], 'logshares']
gf.get_corr_heat_map(df.loc[:, corrcols])

#####Calculate correlation between content (links, videos) and log shares
corrcols = [*groups['media_cols'], 'logshares']
gf.get_corr_heat_map(df.loc[:, corrcols])

#####Calculate correlation between keyword cols and logshares
corrcols = ['num_keywords', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg', 'logshares']
gf.get_corr_heat_map(df.loc[:, corrcols])

####Calculate shares by category in boxplot
#for each in category columns, make entry in new column
for cat in groups['cat_cols']:
    catrows = df.loc[:,cat]==1
    df.loc[catrows,'category'] = cat.split('_')[-1]
    
#plot the distributions of shares by category in box plot
#df['logshares'] = np.log(df.iloc[:,share_col[0]])
plt.figure()
#sns.catplot(x=df.loc[df['category'].notnull(),'category'], y=df[df['category'].notnull()].iloc[:,share_col[0]], kind='violin')
g = sns.catplot(x='category', y='logshares', data=df, kind='box')
g.set_xticklabels(rotation=30)
plt.show()

#####Calculate shares by day of week in boxplot
#for each day in day columns, make entry in new column
for day in groups['day_cols']:
    dayrows = df.loc[:,day]==1
    df.loc[dayrows,'dayofweek'] = day.split('_')[-1]
    
#plot the distributions of shares in box plot by day of week
plt.figure()
g = sns.catplot(x='dayofweek', y='logshares', data=df, kind='box')
g.set_xticklabels(rotation=30)
g.set()
plt.show()

#### Plot log shares grouped by day and category
plt.figure()
g = sns.barplot(x='dayofweek', y='logshares', hue='category', data=df)
g.set_ylim(7)

#####Calculate correlation of lda_cols with log shares correlation matrix
corrcols = [*groups['lda_cols'], 'logshares']
gf.get_corr_heat_map(df.loc[:, corrcols])

#####Calculate correlation of polar_cols with log shares correlation matrix
corrcols = [*groups['polar_cols'], 'logshares']
gf.get_corr_heat_map(df.loc[:, corrcols])

#####Calculate correlation of subj_cols with log shares correlation matrix
corrcols = [*groups['subj_cols'], 'logshares']
gf.get_corr_heat_map(df.loc[:, corrcols])

#### shares by LDA topic and category
df1 = df[[*groups['lda_cols'], 'category', 'logshares']]
df2 = pd.melt(df1, id_vars=['category','logshares'])
plt.figure()
sns.barplot(x='category', y='value', hue='variable', data=df2)
plt.legend(loc='upper right')