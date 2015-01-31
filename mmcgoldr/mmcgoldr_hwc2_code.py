# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:44:21 2015

@author: mmcgoldr
"""

#CLASS 2 HW (DATA WRANGLING WITH PANDAS)
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as ma

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

data_url = 'https://raw.githubusercontent.com/cs109/content/master/imdb_top_10000.txt'
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv(data_url, delimiter='\t', names=names).dropna()

print "Number of rows: %i" % data.shape[0]
data.isnull()
data.head()
data.tail()

#clean data

#example: how to parse text
#dirty = '142 mins.'
#number, text = dirty.split(' ')
#clean = int(number)
#print number

#remove "mins." from runtime, make float
clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
data.head()

#determine the unique genres
unique_genres = set()
for m in data.genres:
    unique_genres.update(g for g in m.split('|'))
unique_genres = sorted(unique_genres)

#make a column for each genre
for genre in unique_genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

#remove year from title
data['title'] = [t[0:-7] for t in data.title]
data.head()

#variable list
print data.columns

#explore quant variables
data[['score', 'runtime', 'votes']].describe()

#classify 0 runtime as null
data.runtime[data.runtime==0].shape[0]
data.runtime[data.runtime==0] = np.nan
data.runtime.describe()

#histograms
plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMBD Rating")

plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")

plt.hist(data.votes, bins=100, color='#cccccc')
plt.xlabel("votes")

plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")

# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]

# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

#sum sums over rows by default (NOT WORKING!!!!!!)
genre_count = np.sort(data[unique_genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})


#axis=1 sums over columns instead
genre_count = data[unique_genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

decade =  (data.year // 10) * 10

tyd = data[['title', 'year']]
tyd['decade'] = decade

tyd.head()

#mean score for all movies in each decade
decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

grouped_scores = data.groupby(decade).score

mean = grouped_scores.mean()
std = grouped_scores.std()

plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.fill_between(decade_mean.index, (decade_mean + std).values,
                 (decade_mean - std).values, color='r', alpha=.2)
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values



#MY DATA EXPLORATION-----------------------------------------------------------
    
#create data frame with film count and means and std of scores and votes by genre
data2 = pd.DataFrame(index=unique_genres, 
                     columns=['film_count','mean_score','std_score','mean_votes','std_votes','mean_runtime','std_runtime'])

data2.film_count = data[unique_genres].sum()[::-1]

for g in unique_genres:
    data2.mean_score[g]=data.score[data[g]==True].mean()
    data2.std_score[g]=data.score[data[g]==True].std()
    data2.mean_votes[g]=data.votes[data[g]==True].mean()
    data2.std_votes[g]=data.votes[data[g]==True].std()
    data2.mean_runtime[g]=data.runtime[data[g]==True].mean()
    data2.std_runtime[g]=data.runtime[data[g]==True].std()

print data2    

#drop genres with less than 10 films
data3 = data2[data2.film_count > 9]

#plot mean score by mean votes
plt.scatter(data3.mean_votes, data3.mean_score, alpha=1, lw=0, color='k')
plt.xlabel("Mean Votes by Genre")
plt.ylabel("Mean Score by Genre")
plt.title('GENRE PERFORMANCE VS REACH')

#identify genre with highest mean score and lowest mean votes
print data3[data3.mean_score > 7.5]

#plot mean score by mean runtime
plt.scatter(data3.mean_runtime, data3.mean_score, alpha=1, lw=0, color='k')
plt.xlabel("Mean Runtime by Genre")
plt.ylabel("Mean Score by Genre")
plt.title('GENRE PERFORMANCE VS LENGTH')

#identify outliers on mean score and mean runtime
print data3[(data3.mean_runtime < 100) & (data3.mean_score > 6.5)]

#INTERESTING FACT 1: Film-Noir tends to have a small, but satisfied, group of followers.
#INTERESTING FACT 2: With two exceptions (Animation, Film-Noir), average genre performance and runtime are positively correlated