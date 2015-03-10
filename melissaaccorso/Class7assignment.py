# -*- coding: utf-8 -*-
"""
Created on Sun Mar 01 12:11:51 2015

@author: melaccor
"""

# importing division from the 'future' release of Python (i.e. Python 3)
from __future__ import division

import pandas
import sqlite3
import numpy as np
from statsmodels.formula.api import logit
from statsmodels.nonparametric import kde
import matplotlib.pyplot as plt
from patsy import dmatrix, dmatrices

# connect to the baseball database
database = r'C:\\Users\\melaccor\\Documents\\SQLite\\lahman2013.sqlite'
conn = sqlite3.connect(database)
sql ="""select a.playerID, a.inducted as inducted, batting.*, pitching.*, fielding.* from
(select playerID, case when avginducted = 0 then 0 else 1 end as inducted from 
(select playerID, avg(case when inducted = 'Y' then 1 else 0 end ) as avginducted from HallOfFame 
where yearid < 2000
group by playerID)) a 
left outer join
(select playerID,  sum(AB) as atbats, sum(H) as totalhits, sum(R) as totalruns, sum(HR) as totalhomeruns, sum(SB) as stolenbases, sum(RBI) as totalRBI, sum(SO) as strikeouts, sum(IBB) as intentionalwalks
from Batting
group by playerID) batting on batting.playerID = a.playerID
left outer join(select playerID, sum(G) as totalgames, sum(SO) as shutouts, sum(sv) as totalsaves, sum(H) as totalhits, sum(er) as earnedruns, sum(so) as strikeouts, sum(WP) as wildpitches, sum(R) as totalruns
from Pitching
group by playerID) pitching on pitching.playerID = a.playerID 
left outer join
(select playerID, sum(G) as games, sum(InnOuts) as timewithouts, sum(PO) as putouts, sum(E) as errors, sum(DP) as doubleplays
from Fielding
group by playerID) fielding on fielding.playerID = a.playerID;"""
# passing the connection and the SQL string to pandas.read_sql.
df = pandas.read_sql(sql, conn)
# closing the connection.
conn.close()

# dropping ALL NaNs in the dataset.
df.dropna(inplace = True)

print df.head()

#removing extra playerID columns
df.drop('playerID',  1, inplace = True)

df.describe()

#logistic regression with multiple predictor variables
model1 = logit('inducted ~ atbats + totalsaves + totalgames + errors + totalhomeruns + totalRBI + doubleplays + intentionalwalks', data = df).fit() # model1 is our fitted model.
print model1.summary()
#p-value for atbats is far from significant, so taking that variable out. Also, taking homeruns out for high p-value

model2 = logit('inducted ~ totalsaves + totalgames + errors + totalRBI + doubleplays + intentionalwalks', data = df).fit() # model1 is our fitted model.
print model2.summary()
#now, doubleplays is the only non-significant variable...taking it out

model3 = logit('inducted ~ totalsaves + totalgames + errors + totalRBI + intentionalwalks', data = df).fit() # model1 is our fitted model.
print model3.summary()
#all factors are significant now on a 10% significance level


#Lets compare the three models using model error rates
#model1
print model1.pred_table()
print 'Model 1 Error rate: {0: 3.0%}'.format(1 - np.diag( model1.pred_table()).sum() / model1.pred_table().sum())
#Model 1 Error rate:  11%
print 'Null Error Rate: {0: 3.0%}'.format(1 - df['inducted'].mean())
#Null Error Rate:  86%
#model2
print model2.pred_table()
print 'Model 2 Error rate: {0: 3.0%}'.format(1 - np.diag( model2.pred_table()).sum() / model2.pred_table().sum())
#Model 2 Error rate:  11%
print 'Null Error Rate: {0: 3.0%}'.format(1 - df['inducted'].mean())
#Null Error Rate:  86%
#model3
print model3.pred_table()
print 'Model 3 Error rate: {0: 3.0%}'.format(1 - np.diag( model3.pred_table()).sum() / model3.pred_table().sum())
#Model 3 Error rate:  11%
print 'Null Error Rate: {0: 3.0%}'.format(1 - df['inducted'].mean())
#Null Error Rate:  86%

#error rates stay the same between these models


#cross validating model3
#splitting out the explanatory features 
colnames = list(df.columns.values)
print colnames
#keeping significant factors
colnames.remove('a.playerID')
colnames.remove('inducted')
colnames.remove('atbats')
colnames.remove('totalhits')
colnames.remove('totalhits')
colnames.remove('totalruns')
colnames.remove('totalruns')
colnames.remove('totalhomeruns')
colnames.remove('stolenbases')
colnames.remove('strikeouts')
colnames.remove('strikeouts')
colnames.remove('shutouts')
colnames.remove('earnedruns')
colnames.remove('wildpitches')
colnames.remove('games')
colnames.remove('timewithouts')
colnames.remove('putouts')
colnames.remove('doubleplays')
print colnames

explanatory_features = colnames
explanatory_df = df[explanatory_features]

#dropping rows with no data.
explanatory_df.dropna(how='all', inplace = True)

#extracting column names
explanatory_colnames = explanatory_df.columns

#doing the same for response
response_series = df.inducted
response_series.dropna(how='all', inplace = True) 

#seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)]
# if there were any, we need to make sure that we only keep indices 
# that are the union of the explanatory and response features post-dropping.

# imputing NaNs with the mean value for that column.  We will 
# go over this in further detail in next week's class.
from sklearn.preprocessing import Imputer
imputer_object = Imputer(missing_values='NaN', strategy='mean', axis=0)
# fitting the object on our data -- we do this so that we can save the 
# fit for our new data.
imputer_object.fit(explanatory_df)
explanatory_df = imputer_object.transform(explanatory_df)

# create a naive Bayes classifier and get it cross-validated accuracy score. 
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

# creating the naive bayes classifier object 
naive_bayes_classifier = MultinomialNB()

# running a cross-validates score on accuracy
accuracy_scores = cross_val_score(naive_bayes_classifier, explanatory_df, response_series, cv=10, scoring='accuracy')

# let's see how accurate the model is, on average.
print accuracy_scores.mean()
#Accuracy has a mean of 0.79 for this model


#considering two features: Inducted and Total Games
def binary_jitter(x, jitter_amount = .05):
    '''
    Add jitter to a 0/1 vector of data for plotting.
    '''
    jitters = np.random.rand(*x.shape) * jitter_amount
    x_jittered = x + np.where(x == 1, -1, 1) * jitters
    return x_jittered
    
# First plot the Inducted / Not Inducted dots vs games to a safe well. Add jitter.
plt.plot(df['totalgames'], binary_jitter(df['inducted'], .1), '.', alpha = .1)
# Now use the model to plot probability of induction vs games
sorted_dist = np.sort(df['totalgames'])
argsorted_dist = list(np.argsort(df['totalgames']))
predicted = model1.predict()[argsorted_dist]
plt.plot(sorted_dist, predicted, lw = 2)

kde_sw = kde.KDEUnivariate(df['totalgames'][df['inducted'] == 1])
kde_nosw = kde.KDEUnivariate(df['totalgames'][df['inducted'] == 0])

kde_sw.fit()
kde_nosw.fit()

plt.plot(kde_sw.support, kde_sw.density, label = 'Inducted')
plt.plot(kde_nosw.support, kde_nosw.density, color = 'red', label = 'Not Inducted')
plt.xlabel('Games')
plt.legend(loc = 'best')



























