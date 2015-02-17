'''
jonblum
2015-02-05
datbos05
class 6 hw
'''

from __future__ import division

import sqlite3
import pandas as pd
import statsmodels.formula.api as smf


conn = sqlite3.connect('/Users/jon/Documents/Code/datbos05/data/lahman2013.sqlite')

sql = '''
SELECT m.nameLast, m.nameFirst, b.playerID, b.yearID, b.yearID - m.birthYear as age, m.height, m.weight, SUM(b.G) as games, SUM(b.AB) as atbats,SUM(H) as hits, SUM("2B") as doubles, SUM("3B") as triples, SUM(HR) as homeRuns, SUM(BB) as walks, SUM(SO) as strikeouts, SUM(SB) as stolenBases, SUM(b.RBI) as rbi, SUM(b.R) as runs
FROM Batting b
LEFT JOIN Master m
ON b.playerID = m.playerID
WHERE yearID > 1920 AND yearID < 2000
GROUP BY b.playerID, b.yearID;
'''

df = pd.read_sql(sql,conn)
conn.close()

# fill in height / weight gaps with averages
mean_height = df.height.mean()
mean_weight = df.weight.mean()
df['height'].fillna(value=mean_height,inplace=True)
df['weight'].fillna(value=mean_weight,inplace=True)


# can drop the rest of the nans (non-batters)
df.dropna(inplace=True)

# potentially-useful derived feature from height and weight
df['bmi'] = (df.weight * 703) / (df.height ** 2)

# get some categorical features around date

df['before_1945'] = 0
df.before_1945[df.yearID < 1945] = 1

df['from_1945_to_1960'] = 0
df.from_1945_to_1960[(df.yearID >= 1945) & (df.yearID<1960)] = 1

df['from_1960_to_1980'] = 0
df.from_1960_to_1980[(df.yearID >= 1960) & (df.yearID<1980)] = 1

# just hits and date
simple_est = smf.ols(formula='runs ~ hits + before_1945 + from_1945_to_1960 + from_1960_to_1980', data=df).fit()

# many more features
complex_est = smf.ols(formula='runs ~ atbats + hits + doubles + triples + homeRuns + walks + strikeouts + stolenBases + bmi + age + before_1945 + from_1945_to_1960 + from_1960_to_1980', data=df).fit()

print simple_est.summary()

# Intercept: -0.27:  If hits were 0 and it were after 1980, there would be effectively no runs.
# This makes sense.
# Hits has a coefficient of .51 -- runs almost exactly half of hits
# For the year categories, all have negative coefficients - negative modifiers, suggesting the
# final implicit year category (> 1980) would be a positive factor


print complex_est.summary()
# Intercept: 6.68 If all else were 0, there would be 6.68 runs.  Obvious, this isn't
# true, but it demonstrates the complexity we've added with our other features.


df['simple_yhat'] = simple_est.predict(df)
df['complex_yhat'] = complex_est.predict(df)

df['simple_residuals'] = df.runs - df.simple_yhat
df['complex_residuals'] = df.runs - df.complex_yhat

# check heteroskedasticity
df.plot(x='runs', y='simple_residuals', kind='scatter')
df.plot(x='runs', y='complex_residuals', kind='scatter')

# for simple, variance increases sharply with hits, so heteroskedastic.
# for complex, less so. Unable to make entirely homoskedastic.

print simple_est.rsquared
# 0.940
print complex_est.rsquared
# 0.969

simple_rmse = (df.simple_residuals ** 2).mean() ** (1/2)
# 7.20
complex_rmse = (df.complex_residuals ** 2).mean() ** (1/2)
# 5.13


# get out-of-sample data

conn = sqlite3.connect('/Users/jon/Documents/Code/datbos05/data/lahman2013.sqlite')

sql = '''
SELECT m.nameLast, m.nameFirst, b.playerID, b.yearID, b.yearID - m.birthYear as age, m.height, m.weight, SUM(b.G) as games, SUM(b.AB) as atbats,SUM(H) as hits, SUM("2B") as doubles, SUM("3B") as triples, SUM(HR) as homeRuns, SUM(BB) as walks, SUM(SO) as strikeouts, SUM(SB) as stolenBases, SUM(b.RBI) as rbi, SUM(b.R) as runs
FROM Batting b
LEFT JOIN Master m
ON b.playerID = m.playerID
WHERE yearID > 2000
GROUP BY b.playerID, b.yearID;
'''

df2 = pd.read_sql(sql,conn)
conn.close()

# fill in height / weight gaps with averages
mean_height = df2.height.mean()
mean_weight = df2.weight.mean()
df2['height'].fillna(value=mean_height,inplace=True)
df2['weight'].fillna(value=mean_weight,inplace=True)


# can drop the rest of the nans (non-batters)
df2.dropna(inplace=True)

# potentially-useful derived feature from height and weight
df2['bmi'] = (df2.weight * 703) / (df2.height ** 2)

# get some categorical features around date

df2['before_1945'] = 0
df2['from_1945_to_1960'] = 0
df2['from_1960_to_1980'] = 0


df2['simple_yhat'] = simple_est.predict(df2)
df2['complex_yhat'] = complex_est.predict(df2)

df2['simple_residuals'] = df2.runs - df2.simple_yhat
df2['complex_residuals'] = df2.runs - df2.complex_yhat

simple_oos_rmse = (df2.simple_residuals ** 2).mean() ** (1/2)
# 6.25
complex_oos_rmse = (df2.complex_residuals ** 2).mean() ** (1/2)
# 4.35
