# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 17:27:16 2015

@author: jeppley
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#BUILDING THE DATA FRAME


data_url = 'http://bit.ly/cs109_imdb'

# passing data url to data object 

names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv('imdb_top_10000.txt', delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head()  # print the first 5 rows


#CLEANING THE DATA FRAME

#Issue 1: Runtime column needs to be numeric

dirty = '142 mins.'
number, text = dirty.split(' ')
clean = int(number)
print number

#Fix the issue for all cases, not just 142 mins, using list comprehension

clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
data.head()

#Splitting a column with a number of values using indicator variables

#determine the unique genres
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|')) #update function create dict
genres = sorted(genres)

#make a column for each genre, ask Teresa about this
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.describe()

#Removing year from the movie title column

print data['title']

data['title'] = [t[0:-7] for t in data.title]

print data['Mystery']

#RUNNING GLOBAL SUMMARIES ON DATA

data[['score', 'runtime', 'year', 'votes']].describe()

#hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])

#probably best to flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan

data.runtime.describe()

#MAKING BASIC PLOTS OF DATA - EXPLORATION OF DATA

# more movies in recent years, but not *very* recent movies (they haven't had time to receive lots of votes yet?)
plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
plt.xlabel("Release Year")

plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")

plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")

#hmm, more bad, recent movies. Real, or a selection bias?

plt.scatter(data.year, data.score, lw=0, alpha=.20, color='r')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")

plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')

# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]

# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

#RUN AGGREGATION FUNCTIONS LIKE SUM OVER SEVERAL ROWS OR COLUMNS

#What genres are most frequent?

#sum sums over rows by default
genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})

print genre_count

#How many genres does a movie have, on average?

#axis=1 sums over columns instead
genre_count = data[genres].sum(axis=1) 
print "Average movie has %.4f genres" % genre_count.mean()
genre_count.describe()

#Exploring grouping properties

decade =  (data.year // 10) * 10

print decade

tyd.head()

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


#compute the scatter in each year

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

#iterating over a groupby statement

for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values
    

#create a 4x6 grid of plots.
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8), 
                         tight_layout=True)

bins = np.arange(1950, 2013, 3)
for ax, genre in zip(axes.ravel(), genres):
    ax.hist(data[data[genre] == 1].year, 
            bins=bins, histtype='stepfilled', normed=True, color='r', alpha=.3, ec='none')
    ax.hist(data.year, bins=bins, histtype='stepfilled', ec='None', normed=True, zorder=0, color='#cccccc')
    
    ax.annotate(genre, xy=(1955, 3e-2), fontsize=14)
    ax.xaxis.set_ticks(np.arange(1950, 2013, 30))
    ax.set_yticks([])
    ax.set_xlabel('Year')
    
    
#EXPLORE THE DATA ON YOUR OWN; TWO INTERESTING FACTS ABOUT DATA
  

#INSIGHT ONE AND PLOT ONE
  
data[['score', 'runtime', 'year', 'votes']].describe()
data.describe()

#Rounding runtime to nearest 10th bucket
runs = np.round((data.runtime / 10)) * 10
print runs

sco = data[['title', 'runtime']]
sco['runs'] = runs

sco.head()

#mean score for all movies with different runtimes

runtime_mean = data.groupby(runs).score.mean()
runtime_mean.name = 'Runtime Mean'
print runtime_mean

#Insight: movies < 120 mins in runtime tend to be more mediocre than movies
# with runtimes > 120 mins

#visual representation of the above finding
plt.plot(runtime_mean.index, runtime_mean.values, 'o-',
        color='r', lw=3, label='Runtime Average')
plt.scatter(data.runtime, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Runtime")
plt.ylabel("Score")
plt.legend(frameon=False)


#INSIGHT TWO AND PLOT TWO

plt.scatter(data.year, data.runtime, lw=0, alpha=.20, color='r')
plt.xlabel("Year")
plt.ylabel("Runtime")

#Insight: Average runtime hasn't changed dramatically over time












#ask in class: iterating over columns to get the mean of cases where the value for the
#column is True

genre_mean = data['Western'].score.mean()

genre_mean = data[data.Western==True].score.mean()
print genre_mean


for column in data.T:
   print data.score.mean(column)

df1.iloc[:,1:3]

data.describe()



