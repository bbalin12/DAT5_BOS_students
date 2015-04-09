# -*- coding: utf-8 -*-
"""
@author: mmcgoldr

"""

#IMPORT PACKAGES---------------------------------------------------------------

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import rcParams as rcp

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.cross_validation import StratifiedKFold as skf
from sklearn.grid_search import GridSearchCV as gscv

import scipy as sp
from scipy import stats as spstat


#LOAD CONFIDENCE INTERVAL FUNCTION---------------------------------------------

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), spstat.sem(a)
    h = se * spstat.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


#SET PANDAS DISPLAY OPTION-----------------------------------------------------

pd.set_option('display.max_rows', None)


#GET DATA----------------------------------------------------------------------

#column names
col = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar',
'chlorides','free_SO2','total_SO2','density','pH','sulphates','alcohol','quality']

#create data frames for red and white sets
dfr = pd.read_csv('C:\Users\mmcgoldr\Dropbox\GA\DataScience\Project\data\winequality-red.csv', sep=';', skiprows=1, names=col)
dfw = pd.read_csv('C:\Users\mmcgoldr\Dropbox\GA\DataScience\Project\data\winequality-white.csv', sep=';', skiprows=1, names=col)

#split data into explanatory and outcome features
dfw_exp = dfw.drop(['quality'], axis=1)
dfw_res = dfw.quality

dfr_exp = dfr.drop(['quality'], axis=1)
dfr_res = dfr.quality


#CONDUCT EDA-------------------------------------------------------------------

#check shape, first and last 10 rows
print dfr.shape
print dfr.head(10)
print dfr.tail(10)

print dfw.shape
print dfw.head(10)
print dfw.tail(10)

#check for missing values (none!)
print dfr.isnull().sum()
print dfw.isnull().sum()

#get outcome distribution (counts and plots)
print dfr.quality.value_counts().sort_index()
print dfw.quality.value_counts().sort_index()

rcp['figure.figsize'] = 6,4
plt.figure()
plt.bar(range(3,9), dfr.quality.value_counts().sort_index(),color='#8A002E', align='center')
plt.axis([2.5, 8.5, 0, 800])
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Observations', fontsize=14)
plt.title('Red', fontsize=14, fontweight='bold', y=1.05)
for i in range(3,9):
    count=dfr.quality.value_counts().sort_index()[i]
    plt.annotate(count, xy=(i,count), xytext=(i-.12,count+20))
    
rcp['figure.figsize'] = 6.7,4
plt.figure()
plt.bar(range(3,10), dfw.quality.value_counts().sort_index(),color=(1,.952,.004), align='center')
plt.axis([2.5, 9.5, 0, 2500])
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Observations', fontsize=14)
plt.title('White', fontsize=14, fontweight='bold', y=1.05)
for i in range(3,10):
    count=dfw.quality.value_counts().sort_index()[i]
    if count > 1000:
        plt.annotate(count, xy=(i,count), xytext=(i-.25,count+50))
    else:
        plt.annotate(count, xy=(i,count), xytext=(i-.12,count+50))
        
#get descriptive stats for explanatory variables
print dfr.describe()
print dfw.describe()

#get plots of mean and median of explanatory variables by quality
red_means_by_qual=pd.DataFrame(dfr.groupby('quality').mean())
red_medians_by_qual=pd.DataFrame(dfr.groupby('quality').median())
white_means_by_qual=pd.DataFrame(dfw.groupby('quality').mean())
white_medians_by_qual=pd.DataFrame(dfw.groupby('quality').median())

min_max=pd.DataFrame(index=range(0,11), columns=['min_val','max_val'])
for i in range(0,11):
    min_max.min_val[i]=min(red_means_by_qual[col[i]].min(),red_medians_by_qual[col[i]].min(),white_means_by_qual[col[i]].min(),white_medians_by_qual[col[i]].min())
    min_max.max_val[i]=max(red_means_by_qual[col[i]].max(),red_medians_by_qual[col[i]].max(),white_means_by_qual[col[i]].max(),white_medians_by_qual[col[i]].max())

red_range=range(3,9)
white_range=range(3,10)

red_color_value='#8A002E'
white_color_value=(1,.952,.004)

for i in range(0,11):
    rcp['figure.figsize'] = 6,4
    plt.figure()
    plt.plot(red_range, red_means_by_qual[col[i]], color=red_color_value, linewidth=3)
    plt.plot(red_range, red_medians_by_qual[col[i]], '--', color=red_color_value, linewidth=3)
    plt.title('Red', fontsize=14, fontweight='bold', y=1.05)
    plt.xlabel('Quality', fontsize=14)
    plt.ylabel(col[i].title(), fontsize=14)
    plt.axis([2.5,8.5,min_max.min_val[i]*.95,min_max.max_val[i]*1.05])

for i in range(0,11):
    rcp['figure.figsize'] = 7,4
    plt.figure()
    plt.plot(white_range, white_means_by_qual[col[i]], color=white_color_value, linewidth=3)
    plt.plot(red_range, red_medians_by_qual[col[i]], '--', color=white_color_value, linewidth=3)
    plt.title('White', fontsize=14, fontweight='bold', y=1.05)
    plt.xlabel('Quality', fontsize=14)
    plt.ylabel(col[i].title(), fontsize=14)
    plt.axis([2.5,9.5,min_max.min_val[i]*.95,min_max.max_val[i]*1.05])

#get Spearman rank correlations for explanatory features
red_corr_rho,red_corr_pval=spstat.spearmanr(dfr_exp)
red_corr_rho=pd.DataFrame(red_corr_rho,index=range(0,11),columns=range(0,11))
red_corr_pval=pd.DataFrame(red_corr_pval,index=range(0,11),columns=range(0,11))
print red_corr_rho
print red_corr_pval

white_corr_rho,white_corr_pval=spstat.spearmanr(dfw_exp)
white_corr_rho=pd.DataFrame(white_corr_rho,index=range(0,11),columns=range(0,11))
white_corr_pval=pd.DataFrame(white_corr_pval,index=range(0,11),columns=range(0,11))
print white_corr_rho
print white_corr_pval

#RANDOM FOREST MODELING: RED---------------------------------------------------

#set iterations
iterations=20

#create empty data frames for prediction results and feature importances
red_results=pd.DataFrame(index=dfr_exp.index, columns=range(0,iterations))
red_features=pd.DataFrame(index=range(0,11), columns=range(0,iterations))

#fit model using StratifiedKFold
rf=rfc(n_estimators=360, max_features=5, criterion='gini')
for j in range(0,iterations):
    folds = skf(dfr_res, 5, shuffle=True)
    for train, test in folds:
        model=rf.fit(dfr_exp.ix[train,], dfr_res[train])
        red_results.ix[test,j] = pd.Series(model.predict(dfr_exp.ix[test,]), index=test, name=[j])
        red_features[j]=pd.Series(model.feature_importances_)
    print j

#write results to file
red_results.to_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/red_results.txt', sep='\t', header=True)
red_features.to_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/red_features.txt', sep='\t', header=True)

#retrieve results as needed
#red_results=pd.read_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/red_results.txt', sep='\t', header=False, names=range(0,iterations))
#red_features=pd.read_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/red_features.txt', sep='\t', header=False, names=range(0,iterations))

#transform results to calculate accuracy, sensitivity (TPR) and precision (PPV)
red_overall_accuracy=pd.Series(0.0, index=range(0,iterations), name='overall_accuracy')
red_kappa=pd.Series(0.0, index=range(0,iterations), name='kappa')
red_tpr=pd.DataFrame(np.zeros((6,iterations),dtype=np.float), index=range(3,9), columns=range(0,iterations))
red_ppv=pd.DataFrame(np.zeros((6,iterations),dtype=np.float), index=range(3,9), columns=range(0,iterations))

red_class_percent=dfr_res.value_counts(normalize=True)
red_class_largest=red_class_percent[red_class_percent==red_class_percent.max()].index

for i in range(0,iterations):
    result=np.array(red_results[i]==dfr_res)
    red_overall_accuracy[i]=result.sum()/round(len(dfr_res),6)
    red_kappa[i]=(red_overall_accuracy[i]-red_class_percent[red_class_largest])/(1-red_class_percent[red_class_largest])
    red_tpr[i]=(pd.Series(result).groupby(dfr_res).sum())/(dfr_res.value_counts())
    red_ppv[i]=(pd.Series(result).groupby(dfr_res).sum())/(red_results[i].value_counts())

#obtain overall accuracy mean and 95% confidence interval
print mean_confidence_interval(red_overall_accuracy, .95)

#obtain kappa mean and 95% confidence interval
print mean_confidence_interval(red_kappa, .95)

#obtain sensitivity (TPR) and precision (PPV) means by class 
print red_tpr.transpose().mean()
print red_ppv.transpose().mean()

#confusion matrix for tolerance=.5 on prediction means
red_results_mean_pred_t50=red_results.transpose().mean().round()
print pd.crosstab(dfr_res, red_results_mean_pred_t50, rownames=['Actual'], colnames=['Predicted'], margins=True)

#obtain feature ranking
red_features_mean_importance=red_features.transpose().mean().sort(ascending=False, inplace=False)
red_features_mean_importance.index=[col[i] for i in red_features_mean_importance.index]
print red_features_mean_importance


#RANDOM FOREST MODELING: WHITE-------------------------------------------------

#set iterations
iterations=20

#create empty data frames for prediction results and feature importances
white_results=pd.DataFrame(index=dfw_exp.index, columns=range(0,iterations))
white_features=pd.DataFrame(index=range(0,11), columns=range(0,iterations))

#fit model using StratifiedKFold
rf=rfc(n_estimators=500, max_features=5, criterion='entropy')
for j in range(0,iterations):
    folds = skf(dfw_res, 5, shuffle=True)
    for train, test in folds:
        model=rf.fit(dfw_exp.ix[train,], dfw_res[train])
        white_results.ix[test,j] = pd.Series(model.predict(dfw_exp.ix[test,]), index=test, name=[j])
        white_features[j]=pd.Series(model.feature_importances_)
    print j

#write results to file
white_results.to_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/white_results.txt', sep='\t', header=True)
white_features.to_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/white_features.txt', sep='\t', header=True)

#retrieve results as needed
#white_results=pd.read_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/white_results.txt', sep='\t', header=False, names=range(0,iterations))
#white_features=pd.read_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/white_features.txt', sep='\t', header=False, names=range(0,iterations))

#transform results to calculate accuracy, sensitivity (TPR) and precision (PPV)
white_overall_accuracy=pd.Series(0.0, index=range(0,iterations), name='overall_accuracy')
white_kappa=pd.Series(0.0, index=range(0,iterations), name='kappa')
white_tpr=pd.DataFrame(np.zeros((7,iterations),dtype=np.float), index=range(3,10), columns=range(0,iterations))
white_ppv=pd.DataFrame(np.zeros((7,iterations),dtype=np.float), index=range(3,10), columns=range(0,iterations))

white_class_percent=dfw_res.value_counts(normalize=True)
white_class_largest=white_class_percent[white_class_percent==white_class_percent.max()].index

for i in range(0,iterations):
    result=np.array(white_results[i]==dfw_res)
    white_overall_accuracy[i]=result.sum()/round(len(dfw_res),6)
    white_kappa[i]=(white_overall_accuracy[i]-white_class_percent[white_class_largest])/(1-white_class_percent[white_class_largest])
    white_tpr[i]=(pd.Series(result).groupby(dfw_res).sum())/(dfw_res.value_counts())
    white_ppv[i]=(pd.Series(result).groupby(dfw_res).sum())/(white_results[i].value_counts())

#obtain overall accuracy mean and 95% confidence interval
print mean_confidence_interval(white_overall_accuracy, .95)

#obtain kappa mean and 95% confidence interval
print mean_confidence_interval(white_kappa, .95)

#obtain sensitivity (TPR) and precision (PPV) means by class 
print white_tpr.transpose().mean()
print white_ppv.transpose().mean()

#confusion matrix for tolerance=.5 on prediction means
white_results_mean_pred_t50=white_results.transpose().mean().round()
print pd.crosstab(dfw_res, white_results_mean_pred_t50, rownames=['Actual'], colnames=['Predicted'], margins=True)

#obtain feature ranking
white_features_mean_importance=white_features.transpose().mean().sort(ascending=False, inplace=False)
white_features_mean_importance.index=[col[i] for i in white_features_mean_importance.index]
print white_features_mean_importance


#RF PARAMETER TUNING: RED------------------------------------------------------

#instantiate model
rf = rfc()

#specify parameter options for number of trees, max number of features and criterion
tree_range = range(20, 1520, 20)
feature_list= ['sqrt',5,7,9,11]
criterion_list=['gini','entropy']
param_grid = dict(n_estimators=tree_range, max_features=feature_list, criterion=criterion_list)
iterations=len(tree_range)*len(feature_list)*len(criterion_list)

#run grid search with F1 scoring
rfgrid_red_unscaled_f1 = gscv(rf, param_grid, cv=5, verbose=5, scoring='f1')
rfgrid_red_unscaled_f1.fit(dfr_exp, dfr_res)

#store results in data frame
rfgrid_red_unscaled_f1_results=pd.DataFrame(index=range(0,iterations),columns=['n_estimators','max_features','criterion','mean_score','all_scores'])
rfgrid_red_unscaled_f1_results.mean_score=[result[1] for result in rfgrid_red_unscaled_f1.grid_scores_]
rfgrid_red_unscaled_f1_results.all_scores=[result[2] for result in rfgrid_red_unscaled_f1.grid_scores_]
rfgrid_red_unscaled_f1_params=[result[0] for result in rfgrid_red_unscaled_f1.grid_scores_]
for i in range(0,iterations):
    rfgrid_red_unscaled_f1_results.n_estimators[i]=rfgrid_red_unscaled_f1_params[i]['n_estimators']
    rfgrid_red_unscaled_f1_results.max_features[i]=rfgrid_red_unscaled_f1_params[i]['max_features']
    rfgrid_red_unscaled_f1_results.criterion[i]=rfgrid_red_unscaled_f1_params[i]['criterion']

#save results to file; retrieve when needed
rfgrid_red_unscaled_f1_results.to_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/rfgrid_red_unscaled_f1_results.txt', sep='\t', header=True)
#rfgrid_red_unscaled_f1_results=pd.read_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/rfgrid_red_unscaled_f1_results.txt', sep='\t', header=False, names=['n_estimators','max_features','criterion','mean_score','all_scores'])

#best estimator
print rfgrid_red_unscaled_f1_results[rfgrid_red_unscaled_f1_results.mean_score==rfgrid_red_unscaled_f1_results.mean_score.max()]

#plot results
rcp['figure.figsize'] = 10,6
plt.figure()
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='gini') & (rfgrid_red_unscaled_f1_results.max_features=='sqrt')].mean_score, 'blue', linewidth=1.2, label='Gini Sqrt(11)')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='gini') & (rfgrid_red_unscaled_f1_results.max_features=='5')].mean_score, 'magenta', linewidth=1.2, label='Gini 5')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='gini') & (rfgrid_red_unscaled_f1_results.max_features=='7')].mean_score, 'cyan', linewidth=1.2, label='Gini 7')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='gini') & (rfgrid_red_unscaled_f1_results.max_features=='9')].mean_score, 'yellow', linewidth=1.2, label='Gini 9')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='gini') & (rfgrid_red_unscaled_f1_results.max_features=='11')].mean_score, 'purple', linewidth=1.2, label='Gini 11')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='entropy') & (rfgrid_red_unscaled_f1_results.max_features=='sqrt')].mean_score, 'red', linewidth=1.2, label='Entropy Sqrt(11)')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='entropy') & (rfgrid_red_unscaled_f1_results.max_features=='5')].mean_score, 'green', linewidth=1.2, label='Entropy 5')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='entropy') & (rfgrid_red_unscaled_f1_results.max_features=='7')].mean_score, 'orange', linewidth=1.2, label='Entropy 7')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='entropy') & (rfgrid_red_unscaled_f1_results.max_features=='9')].mean_score, 'brown', linewidth=1.2, label='Entropy 9')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results[(rfgrid_red_unscaled_f1_results.criterion=='entropy') & (rfgrid_red_unscaled_f1_results.max_features=='11')].mean_score, 'gray', linewidth=1.2, label='Entropy 11')
plt.plot(tree_range, rfgrid_red_unscaled_f1_results.groupby('n_estimators').mean_score.mean(), 'k', linewidth=2,label='Mean')
plt.title('Random Forest Performance: Red Samples', fontsize=14, fontweight='bold', y=1.05)
plt.xlabel('Number of Estimators', fontsize=14)
plt.ylabel('Mean F1 (5-fold CV)', fontsize=14)
plt.legend(fontsize=12, bbox_to_anchor=(1.29, 1.02))
plt.axis([20, 1500, .52, .57])

#mean and max f1 score by max_features and criterion
print rfgrid_red_unscaled_f1_results[rfgrid_red_unscaled_f1_results.criterion=='gini'].groupby(['max_features']).mean_score.mean()
print rfgrid_red_unscaled_f1_results[rfgrid_red_unscaled_f1_results.criterion=='entropy'].groupby(['max_features']).mean_score.mean()
print rfgrid_red_unscaled_f1_results[rfgrid_red_unscaled_f1_results.criterion=='gini'].groupby(['max_features']).mean_score.max()
print rfgrid_red_unscaled_f1_results[rfgrid_red_unscaled_f1_results.criterion=='entropy'].groupby(['max_features']).mean_score.max()


#RF PARAMETER TUNING: RED------------------------------------------------------

#instantiate model
rf = rfc()

#specify parameter options for number of trees, max number of features and criterion
tree_range = range(20, 1520, 20)
feature_list= ['sqrt',5,7,9,11]
criterion_list=['gini','entropy']
param_grid = dict(n_estimators=tree_range, max_features=feature_list, criterion=criterion_list)
iterations=len(tree_range)*len(feature_list)*len(criterion_list)

#run grid search and fit model with F1 scoring
rfgrid_white_unscaled_f1 = gscv(rf, param_grid, cv=5, verbose=5, scoring='f1')
rfgrid_white_unscaled_f1.fit(dfw_exp, dfw_res)
    
#store results in data frame
rfgrid_white_unscaled_f1_results=pd.DataFrame(index=range(0,iterations),columns=['n_estimators','max_features','criterion','mean_score'])
rfgrid_white_unscaled_f1_results.mean_score=[result[1] for result in rfgrid_white_unscaled_f1.grid_scores_]
rfgrid_white_unscaled_f1_params=[result[0] for result in rfgrid_white_unscaled_f1.grid_scores_]
for i in range(0,iterations):
    rfgrid_white_unscaled_f1_results.n_estimators[i]=rfgrid_white_unscaled_f1_params[i]['n_estimators']
    rfgrid_white_unscaled_f1_results.max_features[i]=rfgrid_white_unscaled_f1_params[i]['max_features']
    rfgrid_white_unscaled_f1_results.criterion[i]=rfgrid_white_unscaled_f1_params[i]['criterion']

#save results to file; retrieve when needed
rfgrid_white_unscaled_f1_results.to_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/rfgrid_white_unscaled_f1_results.txt', sep='\t', header=True)
#rfgrid_white_unscaled_f1_results=pd.read_csv('C:/Users/mmcgoldr/Dropbox/GA/DataScience/Project/rfgrid_white_unscaled_f1_results.txt', sep='\t', header=False, names=['n_estimators','max_features','criterion','mean_score'])

#best estimator
print rfgrid_white_unscaled_f1_results[rfgrid_white_unscaled_f1_results.mean_score==rfgrid_white_unscaled_f1_results.mean_score.max()]

#plot results
rcp['figure.figsize'] = 10,6
plt.figure()
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='gini') & (rfgrid_white_unscaled_f1_results.max_features=='sqrt')].mean_score, 'blue', linewidth=1.2, label='Gini Sqrt(11)')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='gini') & (rfgrid_white_unscaled_f1_results.max_features=='5')].mean_score, 'magenta', linewidth=1.2, label='Gini 5')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='gini') & (rfgrid_white_unscaled_f1_results.max_features=='7')].mean_score, 'cyan', linewidth=1.2, label='Gini 7')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='gini') & (rfgrid_white_unscaled_f1_results.max_features=='9')].mean_score, 'yellow', linewidth=1.2, label='Gini 9')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='gini') & (rfgrid_white_unscaled_f1_results.max_features=='11')].mean_score, 'purple', linewidth=1.2, label='Gini 11')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='entropy') & (rfgrid_white_unscaled_f1_results.max_features=='sqrt')].mean_score, 'red', linewidth=1.2, label='Entropy Sqrt(11)')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='entropy') & (rfgrid_white_unscaled_f1_results.max_features=='5')].mean_score, 'green', linewidth=1.2, label='Entropy 5')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='entropy') & (rfgrid_white_unscaled_f1_results.max_features=='7')].mean_score, 'orange', linewidth=1.2, label='Entropy 7')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='entropy') & (rfgrid_white_unscaled_f1_results.max_features=='9')].mean_score, 'brown', linewidth=1.2, label='Entropy 9')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results[(rfgrid_white_unscaled_f1_results.criterion=='entropy') & (rfgrid_white_unscaled_f1_results.max_features=='11')].mean_score, 'gray', linewidth=1.2, label='Entropy 11')
plt.plot(tree_range, rfgrid_white_unscaled_f1_results.groupby('n_estimators').mean_score.mean(), 'k', linewidth=2,label='Mean')
plt.title('Random Forest Performance: White Samples', fontsize=14, fontweight='bold', y=1.05)
plt.xlabel('Number of Estimators', fontsize=14)
plt.ylabel('Mean F1 (5-fold CV)', fontsize=14)
plt.legend(fontsize=12, bbox_to_anchor=(1.29, 1.02))
plt.axis([20, 1500, .465, .505])

#mean and max f1 score by max_features and criterion
print rfgrid_white_unscaled_f1_results[rfgrid_white_unscaled_f1_results.criterion=='gini'].groupby(['max_features']).mean_score.mean()
print rfgrid_white_unscaled_f1_results[rfgrid_white_unscaled_f1_results.criterion=='entropy'].groupby(['max_features']).mean_score.mean()
print rfgrid_white_unscaled_f1_results[rfgrid_white_unscaled_f1_results.criterion=='gini'].groupby(['max_features']).mean_score.max()
print rfgrid_white_unscaled_f1_results[rfgrid_white_unscaled_f1_results.criterion=='entropy'].groupby(['max_features']).mean_score.max()
