# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 17:03:39 2015

@author: Margaret
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:10:31 2015

@author: Margaret
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

import sqlite3
import pandas

# importing statsmodels to run the linear regression
# scikit-learn also has a linear model, but stats is more user friendly output
import statsmodels.formula.api as smf

# connect the baseball database
conn = conn = sqlite3.connect('/Users/Margaret/Desktop/data_science/general_assembly/sqlite/lahman2013.sqlite')

sql = """ 
SELECT yearID, sum(R) as total_runs, sum('2B') as doubles, sum('3B') as triples, 
sum(G_batting) as batgames, 
sum(SO) as strikes, sum(CS) as caught,
sum(BB) as base_balls, sum(IBB) as walks
FROM Batting
WHERE yearID > 1901 and yearID < 2000
GROUP BY yearID
ORDER BY yearID ASC
"""

df = pandas.read_sql(sql, conn)
conn.close()

# drop all NaNs in dataset
df.dropna(inplace = True)

# create variables
df['walk_allowed'] = 0
df.walk_allowed[df.walks > 0] = 1


bin_est = smf.ols(formula='total_runs ~ walk_allowed',data=df).fit()
print bin_est.summary()
# intercept represents all years before 1985
# p value just tells us whether there is a relationship - there is most likely a relationship

df['bin_yhat']=bin_est.predict(df)

plt = df.plot(x='yearID',y='total_runs',title ='Binary Walks' kind='scatter')
plt.plot(df.yearID,df.bin_yhat,color='blue',linewidth=3)


# look at residuals to see if there's heteroskedasticity
df['bin_residuals'] = df.total_runs - df.bin_yhat

plt = df.plot(x = 'total_runs', y='bin_residuals', kind = 'scatter')

# calculating RMSE, ** for exponent in python
RMSE = (((df.bin_residuals)**2).mean()**(1/2))

# understand % by which model deviates from actuals on average
perc_avg_dev = RMSE / df.total_runs.mean()
# string formatting with python
print 'average deviation: {0}%'.format(round(perc_avg_dev*100,1))
