# __Kim Kraunz__
# Class 7 Homework - Logistic Regression


## Introduction
I used the Lahman Baseball Database for all analysis. In this homework I used Logistic Regression to predict Hall of Fame Induction.  I modified the model until all features were statistically significant.  I used K-fold Cross Validation to determine the accuracy of the final model and plotted number of All Star games vs. number of times on the Hall of Fame ballot with a seperation line.

I used the following query to pull features from the Lahman Baseball Database:

```
import sqlite3
import pandas
import numpy
import matplotlib.pyplot as plt

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
Group by i.playerI
'''

df = pandas.read_sql(sql, conn)
conn.close()
```

I  wanted to make sure that my databse imported correctly so I printed out the first 5 rows as well as the column names.
```
pandas.set_option('display.max_columns', None)

df.head()
df.columns
```

####Data Manipulation
I manipulated the data in the following ways:

1. I created a binary variable inducted1 from the inducted variable

2. I created a years_played variable from the years_pitched and years_batted variables

3. Finally, I dropped unneccessary variables (playerID, inducted, years_pitched, years_batter), variables with perfect correlation(total_2B), and variables that did not add to the model (birthCountry, total_post_RBIs) as determined in Homework 9

```
df['inducted1'] = 0
df.inducted1[df.inducted == 'Y'] = 1

df['years_played'] = 0
df.years_played[df.years_pitched >= df.years_batted] = df.years_pitched
df.years_played[df.years_pitched < df.years_batted] = df.years_batted

df.drop(['playerID', 'inducted', 'years_pitched', 'years_batted', 'birthCountry', 'total_2B', 'total_post_RBIs'],  1, inplace = True)
df.head()
```

#### Response and explanatory variables

I defined the response and explanatory variables using the following code:
```
explanatory_features = [col for col in df.columns if col not in ['inducted1']]
explanatory_df = df[explanatory_features]
explanatory_df.dropna(how = 'all', inplace = True)
explanatory_col_names = explanatory_df.columns

response_series = df.inducted1
response_series.dropna(how = 'all', inplace = True)

response_series.index[~response_series.index.isin(explanatory_df.index)]

```

#### Data Cleaning
I cleaned the data by first splitting the explanatory variables in to string and numeric data.  I then filled any Nans in the categorical data with 'Nothing', matched my categorical data to Homework 9, and created dummy variables from the categorical data.  I filled any Nans in the numerical data with the feature median.  Finally, I merged the string and numerial data back together into a Pandas dataframe.

```
# Splits data into strings and numeric features

string_features = explanatory_df.ix[: , explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[: , explanatory_df.dtypes != 'object']

string_features.head()
numeric_features.head()

# fills string Nans with nothing

string_features = string_features.fillna('Nothing')
string_features.teamID.value_counts(normalize = True)

# matched categorical data to HW 9

string_features.teamID[(string_features.teamID != 'LAN') &
(string_features.teamID != 'MIN') & (string_features.teamID != 'ML4') & (string_features.teamID != 'MON') & (string_features.teamID != 'NY1') & (string_features.teamID != 'NYA') & (string_features.teamID != 'NYN') & (string_features.teamID != 'Nothing')] = 'Other'
    
string_features.teamID.value_counts(normalize = True)

# Changes categorical data into dummy variables
def get_binary_values(data_frame):
   """encodes cateogrical features in Pandas.
   """
   all_columns = pandas.DataFrame( index = data_frame.index)
   for col in data_frame.columns:
       data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
       all_columns = pandas.concat([all_columns, data], axis=1)
   return all_columns

encoded_data = get_binary_values(string_features)

from sklearn.preprocessing import Imputer

imputer_object = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

numeric_features.head()

# Merges string and numeric DFs back together

explanatory_df = pandas.concat([numeric_features, encoded_data], axis = 1)
explanatory_df.head()
```
I found that when I first tried to run Logistic Regression to predict Hall of Fame induction, that I got the following error:

LinAlgError: Singular matrix

After spending too much time reading documentation online, I realized that it might be a correlation problem.  I had previously removed features with perfect correlation but I changed the correlation limit from 1.0 to 0.9 in the find_perfect_corr function.  I found that total_hits, total_RBIs, and total_3B had correlations with other features of over 0.9 so I removed them from the dataset.
```
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
```

I then merged the response_series and explanatory_df back together.
```
data= pandas.concat([response_series, explanatory_df], axis = 1)
data.head()
data.describe()
```
####Logistic Regression
I then set up my logistic regression model and ran it.
```

################### Logistic Regression  ##################################

import statsmodels.api as sm

train_cols = data.columns[1:] 
model1 = sm.Logit(data['inducted1'], data[train_cols])

result = model1.fit()
print result.summary()

                           Logit Regression Results                           
==============================================================================
Dep. Variable:              inducted1   No. Observations:                  948
Model:                          Logit   Df Residuals:                      925
Method:                           MLE   Df Model:                           22
Date:                Sat, 07 Mar 2015   Pseudo R-squ.:                  0.3972
Time:                        18:35:01   Log-Likelihood:                -308.15
converged:                      False   LL-Null:                       -511.19
                                        LLR p-value:                 2.302e-72
========================================================================================
                           coef    std err          z      P>|z|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------------
final_year_voted        -0.0163      0.007     -2.259      0.024        -0.030    -0.002
years_voted              0.1254      0.021      6.114      0.000         0.085     0.166
total_stolen_bases    6.853e-05      0.001      0.065      0.948        -0.002     0.002
total_runs               0.0027      0.001      5.060      0.000         0.002     0.004
total_HRs               -0.0027      0.002     -1.671      0.095        -0.006     0.000
total_SOs                0.0007      0.000      1.848      0.065     -3.99e-05     0.001
avg_ERA                 -0.0168      0.090     -0.186      0.852        -0.193     0.160
total_wins               0.0116      0.003      3.472      0.001         0.005     0.018
total_saves              0.0087      0.004      2.224      0.026         0.001     0.016
total_errors             0.0003      0.001      0.393      0.694        -0.001     0.002
total_post_hits         -0.0183      0.015     -1.239      0.215        -0.047     0.011
total_post_HRs           0.1934      0.093      2.084      0.037         0.012     0.375
num_of_allstar_games     0.1538      0.034      4.509      0.000         0.087     0.221
years_played            -0.0442      0.031     -1.406      0.160        -0.106     0.017
teamID_LAN              25.8975     14.259      1.816      0.069        -2.051    53.846
teamID_MIN              25.7056     14.267      1.802      0.072        -2.256    53.668
teamID_ML4              26.7298     14.370      1.860      0.063        -1.436    54.895
teamID_MON              -7.1864   2.17e+07  -3.31e-07      1.000     -4.25e+07  4.25e+07
teamID_NY1              27.3970     14.136      1.938      0.053        -0.310    55.104
teamID_NYA              25.7193     14.138      1.819      0.069        -1.991    53.429
teamID_NYN              25.5268     14.292      1.786      0.074        -2.486    53.539
teamID_Nothing          32.4163     14.367      2.256      0.024         4.258    60.574
teamID_Other            26.4255     14.157      1.867      0.062        -1.322    54.173
========================================================================================
```
I decide to remove features that had a p value greater than 0.1 from the model. Those features included: total_stolen_bases, avg_ERA, total_errors, total_post_hits,. years_played, and teamID_MON.
```
data.drop(['total_stolen_bases', 'avg_ERA', 'total_errors', 'total_post_hits', 'years_played', 'teamID_MON'],  1, inplace = True)

train_cols = data.columns[1:] 
model2 = sm.Logit(data['inducted1'], data[train_cols])
 
result = model2.fit()
print result.summary()

                          Logit Regression Results                           
==============================================================================
Dep. Variable:              inducted1   No. Observations:                  948
Model:                          Logit   Df Residuals:                      931
Method:                           MLE   Df Model:                           16
Date:                Sat, 07 Mar 2015   Pseudo R-squ.:                  0.3938
Time:                        18:40:16   Log-Likelihood:                -309.90
converged:                       True   LL-Null:                       -511.19
                                        LLR p-value:                 1.056e-75
========================================================================================
                           coef    std err          z      P>|z|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------------
final_year_voted        -0.0185      0.007     -2.625      0.009        -0.032    -0.005
years_voted              0.1234      0.020      6.116      0.000         0.084     0.163
total_runs               0.0024      0.000      7.632      0.000         0.002     0.003
total_HRs               -0.0023      0.001     -1.664      0.096        -0.005     0.000
total_SOs                0.0006      0.000      1.736      0.083      -7.9e-05     0.001
total_wins               0.0111      0.003      3.400      0.001         0.005     0.018
total_saves              0.0081      0.004      2.094      0.036         0.001     0.016
total_post_HRs           0.1146      0.067      1.707      0.088        -0.017     0.246
num_of_allstar_games     0.1481      0.033      4.426      0.000         0.083     0.214
teamID_LAN              29.5220     13.960      2.115      0.034         2.160    56.884
teamID_MIN              29.3907     13.950      2.107      0.035         2.049    56.733
teamID_ML4              30.3795     14.065      2.160      0.031         2.813    57.946
teamID_NY1              31.1004     13.822      2.250      0.024         4.010    58.190
teamID_NYA              29.4000     13.834      2.125      0.034         2.286    56.514
teamID_NYN              29.2171     13.975      2.091      0.037         1.826    56.608
teamID_Nothing          36.8127     14.024      2.625      0.009         9.326    64.300
teamID_Other            30.1328     13.845      2.177      0.030         2.998    57.268
========================================================================================
```
After rerunning the model with fewer features, I decided to eliminate any features that had a p value greater than 0.05.  Those features included total_HRs, total_SOs, and total_post_HRs.

```
data.drop(['total_HRs', 'total_SOs', 'total_post_HRs'],  1, inplace = True)

train_cols = data.columns[1:] 
model3 = sm.Logit(data['inducted1'], data[train_cols])
 
result = model3.fit()
print result.summary()

                           Logit Regression Results                           
==============================================================================
Dep. Variable:              inducted1   No. Observations:                  948
Model:                          Logit   Df Residuals:                      934
Method:                           MLE   Df Model:                           13
Date:                Sat, 07 Mar 2015   Pseudo R-squ.:                  0.3858
Time:                        18:44:46   Log-Likelihood:                -313.96
converged:                       True   LL-Null:                       -511.19
                                        LLR p-value:                 3.333e-76
========================================================================================
                           coef    std err          z      P>|z|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------------
final_year_voted        -0.0182      0.007     -2.765      0.006        -0.031    -0.005
years_voted              0.1201      0.020      6.079      0.000         0.081     0.159
total_runs               0.0022      0.000      8.132      0.000         0.002     0.003
total_wins               0.0159      0.002      7.548      0.000         0.012     0.020
total_saves              0.0084      0.004      2.162      0.031         0.001     0.016
num_of_allstar_games     0.1479      0.030      4.866      0.000         0.088     0.207
teamID_LAN              29.3347     13.077      2.243      0.025         3.704    54.965
teamID_MIN              28.8379     13.052      2.209      0.027         3.257    54.419
teamID_ML4              29.6578     13.177      2.251      0.024         3.831    55.485
teamID_NY1              30.5506     12.919      2.365      0.018         5.230    55.871
teamID_NYA              29.0538     12.934      2.246      0.025         3.703    54.404
teamID_NYN              28.9987     13.104      2.213      0.027         3.315    54.682
teamID_Nothing          36.2929     13.117      2.767      0.006        10.584    62.001
teamID_Other            29.5979     12.950      2.286      0.022         4.216    54.980
========================================================================================
```

I decided to try running the model after dropping the teamID's because they had strange coefficients.

```
data.drop(['teamID_LAN', 'teamID_MIN', 'teamID_ML4', 'teamID_NY1', 'teamID_NYA', 'teamID_NYN', 'teamID_Nothing', 'teamID_Other'],  1, inplace = True)

train_cols = data.columns[1:] 
model4 = sm.Logit(data['inducted1'], data[train_cols])
 
result = model4.fit()
print result.summary()

                           Logit Regression Results                           
==============================================================================
Dep. Variable:              inducted1   No. Observations:                  948
Model:                          Logit   Df Residuals:                      942
Method:                           MLE   Df Model:                            5
Date:                Sat, 07 Mar 2015   Pseudo R-squ.:                  0.1919
Time:                        19:38:22   Log-Likelihood:                -413.06
converged:                       True   LL-Null:                       -511.19
                                        LLR p-value:                 1.806e-40
========================================================================================
                           coef    std err          z      P>|z|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------------
final_year_voted        -0.0025      0.000    -12.687      0.000        -0.003    -0.002
years_voted              0.0753      0.017      4.308      0.000         0.041     0.110
total_runs               0.0019      0.000      8.472      0.000         0.001     0.002
total_wins               0.0133      0.002      7.386      0.000         0.010     0.017
total_saves              0.0045      0.004      1.203      0.229        -0.003     0.012
num_of_allstar_games     0.0506      0.023      2.197      0.028         0.005     0.096
========================================================================================

```

I had previously used statsmodel to run the logistic regresion but then imported scikit learn’s LogisticRegression to run a 10-fold cross validation.

```
# Cross-validation

from patsy import dmatrices
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

y, X = dmatrices('inducted1 ~ final_year_voted + years_voted +  total_runs + num_of_allstar_games + total_wins + total_saves + teamID_LAN +  teamID_MIN + teamID_ML4 + teamID_NY1 + teamID_NYA + teamID_NYN + teamID_Nothing + teamID_Other', data, return_type="dataframe")
print X.columns

y = numpy.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model1 = LogisticRegression()
model1.fit(X_train, y_train)

pred_train = model1.predict(X_train)
pred_test = model1.predict(X_test)

pandas.crosstab(y_train, pred_train, rownames=["Actual"], colnames=["Predicted"])

scores = cross_val_score(LogisticRegression(), X, y, scoring='roc_auc', cv=10)
print scores
print scores.mean()
```
The ROC score of the 10 fold cross validated model is .8734.  OK but the Random Forest and Boosting Tree had higher accuracies as determined using the ROC score.

Lastly, I plotted all star games vs. number of years voted for on the Hall of Fame ballot with a separation line showing the classification.

```
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
```
![ScatterwithSeparation](https://github.com/bbalin12/DAT5_BOS_students/blob/master/kimkraunz/Class_7_HW_LogisticRegression/ScatterwithSeparation.png)


#### Conclusions

I used four different logistic regression models to predict Hall of Fame induction.  The model that had least number numerical features but included the teamID binary variables had was the strongest model as determined by the pseudo R squared.  The 10-fold cross-validated ROC score for this model was decent (.8734) but the ROC scores for the Random Forest and Boosting Tree models were higher indicating that they are better models than logistic regression for predicting Hall of Fame induction.

