# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 20:40:11 2015

@author: YihanLi
"""


from __future__ import division


import sqlite3
import pandas
import statsmodels.formula.api as smf

conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = """select yearID, sum(R) as total_runs, sum(H) as total_hits, sum(SB) as stolen_bases, sum(SO) as strikeouts, sum(IBB) as total_intentional_walks
from Batting 
where yearID > 1954
and yearID < 2005
group by yearID
order by yearID ASC"""

df = pandas.read_sql(sql, conn)
conn.close()

df.dropna(inplace = True)

#Model 1

df['post_1995']=0
df.post_1995[df.yearID>1995] = 1
df['from_1985_to_1995']=0
df.from_1985_to_1995[(df.yearID>1985)&(df.yearID<=1995)]=1


bin_est = smf.ols(formula='total_runs~total_hits+from_1985_to_1995+post_1995', data=df).fit()

print bin_est.summary()

print bin_est.rsquared

df['yhat'] = bin_est.predict(df)

plt=df.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df.yearID, df.yhat, color='blue', linewidth=3)


plt = df.plot(x='total_hits', y='total_runs', kind='scatter')
plt.plot(df.total_hits, df.yhat, color='blue', linewidth=3)

df['residuals'] = df.total_runs - df.yhat


#Homoskedastic
plt = df.plot(x='total_hits', y='residuals', kind='scatter')

RMSE = (((df.residuals)**2).mean())** (1/2)

print RMSE

percentage_avg_dev = RMSE /df.total_runs.mean()

print percentage_avg_dev

print 'average deviation:{0}%'.format(round(percentage_avg_dev*100,1))






sb_est = smf.ols(formula = 'total_runs ~ stolen_bases+from_1985_to_1995+post_1995', data=df).fit()

print sb_est.summary()

print sb_est.rsquared

df['sb_yhat']=sb_est.predict(df)

plt=df.plot(x='stolen_bases', y='total_runs', kind='scatter')
plt.plot(df.stolen_bases, df.sb_yhat, color='blue', linewidth=3)

df['sb_residuals']=df.total_runs - df.sb_yhat

#Almost homo
plt = df.plot(x='total_runs', y='sb_residuals', kind='scatter')


sb_RMSE = (((df.sb_residuals)**2).mean())** (1/2)

print sb_RMSE

percentage_avg_dev = sb_RMSE /df.total_runs.mean()

print percentage_avg_dev

print 'average deviation:{0}%'.format(round(percentage_avg_dev*100,1))

print sb_est.rsquared


conn = sqlite3.connect('/Users/YihanLi/Documents/SQLite/lahman2013.sqlite')

sql = """select yearID, sum(R) as total_runs, sum(H) as total_hits, sum(SB) as stolen_bases, sum(SO) as strikeouts, sum(IBB) as total_intentional_walks
from Batting 
where yearID >= 2005
group by yearID
order by yearID ASC"""

df_post_2005 = pandas.read_sql(sql, conn)
conn.close()


df_post_2005['post_1995']=1
df_post_2005['from_1985_to_1995']=0

df_post_2005['yhat']=bin_est.predict(df_post_2005)
df_post_2005['large_yhat']=sb_est.predict(df_post_2005)

df_post_2005['bin_residuals']=df_post_2005.total_runs -df_post_2005.yhat

df_post_2005['sb_residuals']=df_post_2005.total_runs-df_post_2005.large_yhat



RMSE_bin = (((df_post_2005.bin_residuals)**2).mean())** (1/2)



RMSE_sb = (((df_post_2005.sb_residuals)**2).mean())** (1/2)


print 'average deviation for first model:{0}'.format(round(RMSE_bin,4))
print 'average deviation for second model:{0}'.format(round(RMSE_sb,4))

# The first model would be a better model with higher r squared and smaller RMSE




plt = df_post_2005.plot(x='yearID', y='total_runs', kind='scatter')
plt.plot(df_post_2005.yearID, df_post_2005.yhat, color='blue', linewidth=3)
plt.plot(df_post_2005.yearID, df_post_2005.large_yhat, color='red', linewidth=3)












