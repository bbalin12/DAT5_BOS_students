# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 19:44:30 2015

@author: garauste
"""

import pandas
import sqlite3
from sklearn import preprocessing
import numpy as np

# Put in seting to print out all columns
pandas.set_option('display.max_columns',None)

# Open the connection
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

query ='''
select m.birthState, b.total_salaries, b.total_runs
from MASTER m 
inner join 
(
select b.playerID, sum(s.salary) as total_salaries, sum(b.R) as total_runs
from batting b 
inner join salaries s
on b.playerID = s.playerID
where b.R is not null and s.yearID < 2000
group by b.playerID
) b
on m.playerID = b.playerID
where b.total_runs >10
group by m.birthState
order by total_runs desc
'''


df= pandas.read_sql(query, conn)
conn.close()

df.head()

# Scale the data
data = df[['total_salaries','total_runs']]
scaler = preprocessing.StandardScaler()
scaler.fit(data)
data = pandas.DataFrame(scaler.transform(data), columns = data.columns)

## Alter plot size
from pylab import rcParams
rcParams['figure.figsize'] = 10,5

# plot the data
plt = df.plot(x='total_salaries',y='total_runs',kind='scatter')

# Annotating with team names
for i, txt in enumerate(df.birthState):
    plt.annotate(txt, (df.total_salaries[i],df.total_runs[i]))
## Some States pay produce players who have extremely high salaries and high career runs while some produce players that earn a lot but don't score a lot of runs


######
# K-Means
######

from sklearn.cluster import KMeans
import matplotlib.pylab as plt
kmeans_est = KMeans(n_clusters = 3)

kmeans_est.fit(data)
labels = kmeans_est.labels_

plt.scatter(df.total_salaries,df.total_runs, s=60, c=labels)
# Reasonable CLustering - The two highest performing states have their own Cluster
# however some of the states which have few runs but quite a lot of expense are 
# clustered with states that are performing relatively well

##############
## DBSCAN
##############

from sklearn.cluster import DBSCAN

# getting around a bug that doesn't let you fit to a dataframe
dbsc = DBSCAN().fit(np.array(data))
labels = dbsc.labels_

plt.scatter(df.total_salaries,df.total_runs, s=60, c=labels)
# DB Scan is even less effective than K-Means clustering

############
## Hierarchial Clustering 
############
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, fclusterdata

distanceMatrix = pdist(data)

# adjust color_threshold to get the number of cluster you want
dend = dendrogram(linkage(distanceMatrix, method='complete'),
                  color_threshold = 1,
                  leaf_font_size = 10,
                  labels = df.birthState.tolist())
                  
# Notice I moved the threshold to 4 so I can get 3 clusters
dend = dendrogram(linkage(distanceMatrix, method='complete'),
                  color_threshold = 4,
                  leaf_font_size = 10,
                  labels = df.birthState.tolist())                  

# get the cluster assignments                  
assignments = fcluster(linkage(distanceMatrix, method= 'complete'),4,'distance')              

# create a dataframe with cluster assignments and names
cluster_output = pandas.DataFrame({'team':df.birthState.tolist(),'cluster':assignments})

# plot the results
plt.scatter(df.total_salaries,df.total_runs,s=60, c=cluster_output.cluster)

# let's make the plot prettier with better assignments
colors = cluster_output.cluster
colors[colors==1]='b'
colors[colors==2]='g'
colors[colors==3]='r'

plt.scatter(df.total_salaries,df.total_runs,s=100,c=colors,lw=0)
 
 
####
# PCA
####

import sqlite3
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import ensemble
import numpy
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt



# including our functions from last week up here for use. 
def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df
#

def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns
#
def find_zero_var(df):
    """finds columns in the dataframe with zero variance -- ie those
        with the same value in every observation.
    """   
    toKeep = []
    toDelete = []
    for col in df:
        if len(df[col].value_counts()) > 1:
            toKeep.append(col)
        else:
            toDelete.append(col)
        ##
    return {'toKeep':toKeep, 'toDelete':toDelete} 
##
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
        perfect_corr = corrMatrix[col][abs(numpy.round(corrMatrix[col],10)) == 1.00].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
    toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}
### 



conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')
sql = '''
select coalesce(a.nameFirst ,"") || coalesce(a.nameLast,"") as PlayerName, hfi.inducted,
a.playerID, a.birthState, b.*, c.*,d.*
from hall_of_fame_inductees hfi
left outer join master a
on a.playerID = hfi.playerID
left outer join 
(
select sum(b.G) as Games, sum(b.H) as Hits, sum(b.AB) as At_Bats, sum(b.HR) as Homers,
b.playerID from Batting b group by b.playerID
) b
on hfi.playerID = b.playerID
left outer join 
(
select d.playerID, sum(d.A) as Fielder_Assists, sum(d.E) as Field_Errors, 
sum(d.DP) as Double_Plays from Fielding d group by d.playerID 
) c
on hfi.playerID = c.playerID
left outer join
(
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws, sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) d 
on hfi.playerID = d.playerID;
'''

df = pandas.read_sql(sql, conn)
conn.close()

df.head(10)
df.columns

# dropping duplicate playerID columns
df.drop('playerID',  1, inplace = True)

#############
## repeating the same preprocessing from the previous lesson
############

## splitting out the explanatory features 
explanatory_features = [col for col in df.columns if col not in ['PlayerName', 'inducted']]
explanatory_df = df[explanatory_features]

# dropping rows with no data.
explanatory_df.dropna(how='all', inplace = True) 

# extracting column names 
explanatory_colnames = explanatory_df.columns

## doing the same for response
response_series = df.inducted
response_series.dropna(how='all', inplace = True) 

## seeing which explanatory feature rows got removed.  Looks like none.
response_series.index[~response_series.index.isin(explanatory_df.index)]

### now, let's seperate the numeric explanatory data from the string data
string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']


# that are all NANs, as they will show up as all 'Nothing' when we start binning or look for features with no variation)
string_features = string_features.fillna('Nothing')
# cleaning up string features
string_features = cleanup_data(string_features)
# binarizing string features 
encoded_data = get_binary_values(string_features)
## imputing features
imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features),
                                    columns = numeric_features.columns)

## pulling together numeric and encoded data.
explanatory_df = pandas.concat([numeric_features, encoded_data],axis = 1)
explanatory_df.head()


#now, let's find features with no variance 
no_variation = find_zero_var(explanatory_df)
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)


############
## PCA
############

from sklearn.decomposition import PCA

pca = PCA(n_components = 6)
pca.fit(explanatory_df)

# extract the components
pca_df = pandas.DataFrame(pca.transform(explanatory_df))

## plotting the first two principal components
pca_df.plot(x=0, y=1, kind = 'scatter')


variance_df = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component':
    pca_df.columns.tolist()})
    
# adding one to pricnipal componetns (since there is no 0th compeonet)
variance_df['principal component'] = variance_df['principal component'] + 1
variance_df.plot(x='principal component', y = 'variance')
#  looks like variance stops getting explained after first two components 

pca_df_small = pca_df.ix[:,0:1]

# getting a cross val score of transformed data
rf = ensemble.RandomForestClassifier(n_estimators = 500)
roc_scores_rf_pca = cross_val_score(rf, pca_df_small, response_series, cv = 10,
                                    scoring = 'roc_auc')
                                    
print roc_scores_rf_pca.mean() 
# 74% accuracy 

roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv = 10,
                                scoring = 'roc_auc') 
print roc_scores_rf.mean() 
# PCA created significant information loss in this case

############################
# Support Vector Machines 
############################

from sklearn.svm import SVC

# first running the quadratic kernel with PCA
svm = SVC(kernel = 'poly')  

roc_scores_svm = cross_val_score(svm, explanatory_df, response_series, cv=10,
                                 scoring='roc_auc')                              
print roc_scores_svm.mean()                                 
# Worse than PCA

# try with PCA
roc_scores_svm_pca = cross_val_score(svm, pca_df_small, response_series, cv=10, 
                                   scoring = 'roc_auc')
print roc_scores_svm_pca.mean()                               

# let's do a grid search
param_grid = dict(kernel = ['linear','poly','rbf','sigmoid'])    

svm_grid = GridSearchCV(svm, param_grid, cv=10, scoring='roc_auc')
svm_grid.fit(explanatory_df,response_series)
best_estimator = svm_grid.best_estimator_
print best_estimator.kernel

# Linear is the best estimator score won
print svm_grid.best_score_                                   
# best estimator was 77% - just below RFs
# Note: SVMs are more accurate than RFs with trending data! 

####################################################
############# Out of Sample Testing ################
####################################################


conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')
# new query to pull data post 2000 
sql = '''
select coalesce(a.nameFirst ,"") || coalesce(a.nameLast,"") as PlayerName, hfi.inducted,
a.playerID, a.birthState, b.*, c.*,d.*
from hall_of_fame_inductees_post2000 hfi
left outer join master a
on a.playerID = hfi.playerID
left outer join 
(
select sum(b.G) as Games, sum(b.H) as Hits, sum(b.AB) as At_Bats, sum(b.HR) as Homers,
b.playerID from Batting b group by b.playerID
) b
on hfi.playerID = b.playerID
left outer join 
(
select d.playerID, sum(d.A) as Fielder_Assists, sum(d.E) as Field_Errors, 
sum(d.DP) as Double_Plays from Fielding d group by d.playerID 
) c
on hfi.playerID = c.playerID
left outer join
(
select b.playerID, sum(b.ERA) as Pitcher_Earned_Run_Avg, sum(b.W) as Pitcher_Ws, sum(b.SHO) as Pitcher_ShutOuts, sum(b.SO) as Pitcher_StrikeOuts,
sum(b.HR) as HR_Allowed, sum(b.CG) as Complete_Games 
from Pitching b 
group by b.playerID
) d 
on hfi.playerID = d.playerID;
'''

df_post2000 = pandas.read_sql(sql,conn)
conn.close()

# Dropping duplicate playerIDs columns
df_post2000.drop('playerID',1,inplace=True)
df_post_colnames = df_post2000.columns

#########
# Preprocesing
#########

## Split out data by explanatory features 
explan_features_post2000 = [col for col in df_post_colnames if col not in ('PlayerName','inducted')]
df_explanatory_post2000 = df_post2000[explan_features_post2000]

# drop rows 
df_explanatory_post2000.dropna(how='all',inplace=True)

# create resposne variable
response_series_post2000 = df_post2000.inducted
response_series_post2000.dropna(how='all',inplace=True)

# seeing which explanatory feature rows got removed 
response_series_post2000.drop(response_series_post2000.index[~response_series_post2000.index.isin(df_explanatory_post2000.index)],inplace=True)

# Create Categorical and Numeric Datasets
string_features_post2000 = df_explanatory_post2000.ix[:,df_explanatory_post2000.dtypes == 'object']
numeric_features_post2000 = df_explanatory_post2000.ix[:,df_explanatory_post2000.dtypes != 'object']

# Determine if categorical data requires binning
string_features_post2000 = string_features_post2000.fillna('Nothing')

# Use function defined earlier to clean the data
string_features_post2000 = cleanup_data(string_features_post2000)

# let's verify if the replacement happened
string_features_post2000.birthState.value_counts(normalize = True)

# Let's encode the categorical variables
# use function 

## Prior to binning need to check to see if unique features are included
## here that are not in pre-2000 data
unique_states = list(set(e for e in string_features.birthState))
unique_states_post2000 = list(set(e for e in string_features_post2000.birthState))
extra_states = list(set(e for e in unique_states_post2000 if e not in unique_states))

# Check string features post 2000
string_features_post2000.birthState.index[~string_features_post2000.index.isin(unique_states)]
# There are a few features that do not occur in the pre-2000 dataset

# Creating Dummy Categorical variables
encoded_data_post2000 = get_binary_values(string_features_post2000)

# Resetting the index
encoded_data_post2000.reset_index(inplace=True)

## Removing extra columns introduced by reset index function
del encoded_data_post2000['index']
#del encoded_data_post2000['level_0']


#######
## Now to check the numeric features
#######

# Use imputed to fill missing values
imputer_object = Imputer(missing_values = 'NaN', strategy = 'median', axis=0)
imputer_object.fit(numeric_features_post2000)
numeric_features_post2000 = pandas.DataFrame(imputer_object.transform(numeric_features_post2000),
                                    columns = numeric_features_post2000.columns)

# Merging encoded and numeric                                    
new_df_post_2000 = pandas.concat([numeric_features_post2000, encoded_data_post2000],axis=1)
new_df_post_2000.head()

# Check to see if there are any columns missing from the new data set that are in the old data
missing_columns=[e for e in explanatory_df.columns if e not in new_df_post_2000.columns]
other_columns = [e for e in new_df_post_2000.columns if e not in explanatory_df.columns]

# Create dummies for missing columns 
for e in missing_columns:
    new_df_post_2000[e] = 0

# Scale the data 
new_df_post_2000= pandas.DataFrame(scaler.transform(new_df_post_2000),columns = 
new_df_post_2000.columns)

#########
# PCA Post 2000
#########

pca = PCA(n_components = 6)
pca.fit(new_df_post_2000)

# extract the components
pca_df_post_2000 = pandas.DataFrame(pca.transform(new_df_post_2000))

## plotting the first two principal components
pca_df_post_2000.plot(x=0, y=1, kind = 'scatter')


variance_df_post_2000 = pandas.DataFrame({'variance': pca.explained_variance_, 'principal component':
    pca_df.columns.tolist()})
    
# adding one to pricnipal componetns (since there is no 0th compeonet)
variance_df_post_2000['principal component'] = variance_df_post_2000['principal component'] + 1
variance_df_post_2000.plot(x='principal component', y = 'variance')
#  looks like variance stops getting explained after first two components 

pca_df_small_post_2000 = pca_df.ix[:,0:1]

# getting a cross val score of transformed data
rf = ensemble.RandomForestClassifier(n_estimators = 500)
roc_scores_rf_pca_post_2000 = cross_val_score(rf, pca_df_small_post_2000, response_series_post2000, cv = 10, scoring = 'roc_auc')
                                    
print roc_scores_rf_pca.mean() 
# 74% accuracy 

roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv = 10,
                                scoring = 'roc_auc') 
print roc_scores_rf.mean() 
# PCA created significant information loss in this case