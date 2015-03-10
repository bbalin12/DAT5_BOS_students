# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 17:17:23 2015

@author: jkraunz
"""

import numpy
import pandas
import sqlite3
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

pandas.set_option('display.max_columns', None)


conn = sqlite3.connect('/Users/jkraunz/Documents/SQLite/lahman2013.sqlite.crdownload')

sql= '''
Select i.*, count(yearID) as num_of_allstar_games
FROM
(Select f.*, birthCountry
FROM
(Select d.*, e.teamID
FROM
(Select c.*, sum(H) as total_post_hits, sum(HR) as total_post_HRs, sum(RBI) as total_post_RBIs
FROM
(Select a.*, sum(E) as total_errors
FROM
(SELECT m.*,
sum(SO) as total_SOs, avg(ERA) as avg_ERA, sum(W) as total_wins, sum(SV) as total_saves, count(YearID) as years_pitched
FROM
(select h.*, sum(RBI) as total_RBIs, sum(SB) as total_stolen_bases, sum(R) as total_runs, sum(H) as total_hits, count(yearID) as years_batted, sum(HR) as total_HRs, sum('2B') as total_2B, sum('3B') as total_3B
FROM 
(SELECT playerID, max(yearID) as final_year_voted, count(yearID) as years_voted, inducted
FROM HallofFame 
Where yearID < 2000
GROUP BY playerID) h
LEFT JOIN Batting b on h.playerID = b.playerID
GROUP BY h.playerID) m
LEFT JOIN Pitching p on m.playerID = p.playerID
group by m.playerID) a
LEFT JOIN Fielding f on a.playerID = f.playerID
GROUP BY a.playerID) c
Left Join BattingPost bp on c.playerID = bp.playerID
Group By c.playerID) d
Left Join dominant_team_per_player e on d.playerID = e.playerID
Group by d.playerID) f
Left Join Master g on f.playerID = g.playerID
Group by f.playerID) i
Left Join num_of_allstar j on i.playerID = j.playerID
Group by i.playerID
'''

df = pandas.read_sql(sql, conn)
conn.close()

df.head()

df.columns


df['inducted1'] = 0
df.inducted1[df.inducted == 'Y'] = 1

df['years_played'] = 0
df.years_played[df.years_pitched >= df.years_batted] = df.years_pitched
df.years_played[df.years_pitched < df.years_batted] = df.years_batted

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'birthCountry', 'total_2B', 'total_post_RBIs'],  1, inplace = True)

df.head()
df.describe()

# Set up explanatory and response features

explanatory_features = [col for col in df.columns if col not in ['inducted1']]
explanatory_df = df[explanatory_features]
explanatory_df.dropna(how = 'all', inplace = True)
explanatory_col_names = explanatory_df.columns

response_series = df.inducted1
response_series.dropna(how = 'all', inplace = True)
response_series.index[~response_series.index.isin(explanatory_df.index)]

response_series.describe()
explanatory_df.describe()
# Splits data into strings and numeric features

string_features = explanatory_df.ix[: , explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[: , explanatory_df.dtypes != 'object']

string_features.head()
numeric_features.head()

string_features.describe()
numeric_features.describe()

# fills string Nans with nothing

string_features = string_features.fillna('Nothing')
string_features.teamID.value_counts(normalize = True)

string_features.describe()
 

# matches categorical data to HW 10

string_features.teamID[(string_features.teamID != 'LAN') &
(string_features.teamID != 'MIN') & (string_features.teamID != 'ML4') & (string_features.teamID != 'MON') & (string_features.teamID != 'NY1') & (string_features.teamID != 'NYA') & (string_features.teamID != 'NYN') & (string_features.teamID != 'Nothing')] = 'Other'
    
string_features.teamID.value_counts(normalize = True)
string_features.describe()


def get_binary_values(data_frame):
   """encodes cateogrical features in Pandas.
   """
   all_columns = pandas.DataFrame( index = data_frame.index)
   for col in data_frame.columns:
       data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
       all_columns = pandas.concat([all_columns, data], axis=1)
   return all_columns

encoded_data = get_binary_values(string_features)

string_features.describe()

from sklearn.preprocessing import Imputer

imputer_object = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

numeric_features.head()
numeric_features.describe()

# Merges string and numeric DFs back together

explanatory_df = pandas.concat([numeric_features, encoded_data], axis = 1)
explanatory_df.head()

explanatory_df.describe()


def find_perfect_corr(df):
    """finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict 
        that includes which columns to drop so that each remaining column
        is independent
    """  
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  numpy.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(numpy.round(corrMatrix[col],10)) >= .9].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
    toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}
find_perfect_corr(explanatory_df)
    
explanatory_df.drop(['total_hits', 'total_RBIs', 'total_3B'], 1, inplace = True)

explanatory_df.describe()

data= pandas.concat([response_series, explanatory_df], axis = 1)
data.head()
data.describe()


################### Logistic Regression  ##################################

import statsmodels.api as sm

train_cols = data.columns[1:] 
model1 = sm.Logit(data['inducted1'], data[train_cols])

result = model1.fit()
print result.summary()

# Decide to remove total_stolen_bases, avg_ERA, total_errors, total_post_hits, years_played, 
# teamID_MON

data.drop(['total_stolen_bases', 'avg_ERA', 'total_errors', 'total_post_hits', 'years_played', 'teamID_MON'],  1, inplace = True)

train_cols = data.columns[1:] 
model2 = sm.Logit(data['inducted1'], data[train_cols])
 
result = model2.fit()
print result.summary()

# Decided to drop total_HRs, total_SOs, total_post_HR
data.drop(['total_HRs', 'total_SOs', 'total_post_HRs'],  1, inplace = True)

train_cols = data.columns[1:] 
model3 = sm.Logit(data['inducted1'], data[train_cols])
 
result = model3.fit()
print result.summary()

# Decided to drop the teamID's because they had strange coefficients
data.drop(['teamID_LAN', 'teamID_MIN', 'teamID_ML4', 'teamID_NY1', 'teamID_NYA', 'teamID_NYN', 'teamID_Nothing', 'teamID_Other'],  1, inplace = True)

train_cols = data.columns[1:] 
model4 = sm.Logit(data['inducted1'], data[train_cols])
 
result = model4.fit()
print result.summary()


# Cross-validation

y, X = dmatrices('inducted1 ~ final_year_voted + years_voted +  total_runs + num_of_allstar_games + total_wins + total_saves + ', data, return_type="dataframe")
print X.columns

y = numpy.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model1 = LogisticRegression()
model1.fit(X_train, y_train)

pred_train = model1.predict(X_train)
pred_test = model1.predict(X_test)

pandas.crosstab(y_train, pred_train, rownames=["Actual"], colnames=["Predicted"])

scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean() # .8639
##################################################################################


# Scatter plot of number of all star games vs. number of years on the hall of fame ballot

logit_pars = result.params
intercept = -logit_pars[0] / logit_pars[2]
slope = -logit_pars[1] / logit_pars[2]

allstar_in = df['num_of_allstar_games'][df['inducted1'] == 1]
allstar_noin = df['num_of_allstar_games'][df['inducted1'] == 0]
voted_in = df['years_voted'][df['inducted1'] == 1]
voted_noin = df['years_voted'][df['inducted1'] == 0]
plt.figure(figsize = (12, 8))
plt.plot(voted_in, allstar_in, '.', mec = 'purple', mfc = 'None', 
         label = 'Inducted')
plt.plot(voted_noin, allstar_noin, '.', mec = 'orange', mfc = 'None', 
         label = 'Not inducted')
plt.plot(numpy.arange(0, 25, 1), intercept + slope * numpy.arange(0, 25, 1) / 100.,
         '-k', label = 'Separating line')
plt.ylim(0, 20)
plt.xlabel('Number of All Star games')
plt.ylabel('Number of years on the Hall of Fame ballot')
plt.legend(loc = 'best')
