# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import graphfuncs as gf
import os.path
import prepare_data as prep

#df = pd.read_csv('OnlineNewsPopularity.csv')
df = prep.prepare_data('OnlineNewsPopularity.csv')

timedelta_col = [1]
word_cols = [2, 3, 4, 5, 6, 11]
media_cols = [7, 8, 9, 10]
cat_cols = [13, 14, 15, 16, 17, 18]
keyword_cols = [12, 19, 20, 21, 22, 23, 24, 25, 26, 27]
ref_cols = [28, 29, 30]
day_cols = [31, 32, 33, 34, 35, 36, 37] #sat 36 and sun 37 cols are empty
lda_cols = [39, 40, 41, 42, 43]
subj_cols = [44, 56, 58]
polar_cols = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 59]
share_col = [60]


#### Strip white space from column names and add log shares
df.columns = df.columns.str.strip()
df['logshares'] = np.log(df['shares'])
logshare_col = [df.columns.get_loc('logshares')]
                
##### Plot timedelta column in histogram
#plt.figure()
#g = sns.distplot(df['timedelta'], bins=30, kde=False)
#plt.show()
gf.get_dist_graph(df, 'timedelta')

##### Make scatter plot of timedelta and log shares
plt.figure()
g = sns.regplot(x=df.loc[df['timedelta']<30, 'timedelta'], y=df.loc[df['timedelta']<30, 'logshares'])
plt.show()

##### Make scatter plot of timedelta and log shares
plt.figure()
g = sns.regplot(x=df['timedelta'], y=df['logshares'])
plt.show()

#####Calculate correlation between time since publ and log shares
timecorr = np.corrcoef(df.iloc[:,timedelta_col], df.iloc[:,logshare_col], rowvar=False)
print(timecorr)


#####Calculate correlation of word_cols with log shares correlation matrix
corrcols = word_cols + logshare_col
#wordcorr = np.corrcoef(df.iloc[:, corrcols], rowvar=False)
#plt.figure()
#flatvals = wordcorr.flatten()
#flatvals.sort()
#sns.heatmap(wordcorr, vmax=flatvals[-(len(wordcorr)+1)], vmin=flatvals[0], xticklabels=df.columns[corrcols], yticklabels=df.columns[corrcols])
gf.get_corr_heat_map(df.iloc[:, corrcols])

#####Calculate correlation between content (links, videos) and log shares
corrcols = media_cols + logshare_col
#mediacorr = np.corrcoef(df.iloc[:, corrcols], rowvar=False)
#plt.figure()
#flatvals = mediacorr.flatten()
#flatvals.sort()
gf.get_corr_heat_map(df.iloc[:, corrcols])

####Calculate shares by category in boxplot
#for each in category columns, make entry in new column
for cat in cat_cols:
    catrows = df.iloc[:,cat]==1
    df.loc[catrows,'category'] = df.columns[cat].split('_')[-1]
    
#plot the distributions of shares by category in box plot
#df['logshares'] = np.log(df.iloc[:,share_col[0]])
plt.figure()
#sns.catplot(x=df.loc[df['category'].notnull(),'category'], y=df[df['category'].notnull()].iloc[:,share_col[0]], kind='violin')
g = sns.catplot(x='category', y='logshares', data=df, kind='box')
g.set_xticklabels(rotation=30)
plt.show()

#####Calculate shares by day of week in boxplot
#for each in category columns, make entry in new column
for day in day_cols:
    dayrows = df.iloc[:,day]==1
    df.loc[dayrows,'dayofweek'] = df.columns[day].split('_')[-1]
    
#plot the distributions of shares in box plot by day of week
#df['logshares'] = np.log(df.iloc[:,share_col[0]])
plt.figure()
#sns.catplot(x=df.loc[df['category'].notnull(),'category'], y=df[df['category'].notnull()].iloc[:,share_col[0]], kind='violin')
g = sns.catplot(x='dayofweek', y='logshares', data=df, kind='box')
g.set_xticklabels(rotation=30)
g.set()
plt.show()

#### Plot log shares grouped by day and category
plt.figure()
g = sns.barplot(x='dayofweek', y='logshares', hue='category', data=df)
g.set_ylim(7)

#####Calculate correlation of lda_cols with log shares correlation matrix
corrcols = lda_cols + logshare_col
gf.get_corr_heat_map(df.iloc[:, corrcols])

#####Calculate correlation of polar_cols with log shares correlation matrix
corrcols = polar_cols + logshare_col
gf.get_corr_heat_map(df.iloc[:, corrcols])

#####Calculate correlation of subj_cols with log shares correlation matrix
corrcols = subj_cols + logshare_col
gf.get_corr_heat_map(df.iloc[:, corrcols])

#### shares by LDA topic and category
df1 = df[['LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','category','logshares']]
df2 = pd.melt(df1, id_vars=['category','logshares'])
plt.figure()
sns.barplot(x='category', y='value', hue='variable', data=df2)