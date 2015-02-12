# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:06:58 2015

@author: megan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

data_url = 'http://bit.ly/cs109_imdb'

####################
# Build a DataFrame
####################

# tab separated data
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv(data_url, delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head()  # print the first 5 rows

######################
# Clean the DataFrame
######################

# As it stands the data has several problems with it
# The runtime column describes a number, but is stored as a string
# The genres column is not atomic -- it aggregates several genres together. This makes it hard, for example, to extract which movies are Comedies.
# The movie year is repeated in the title and year column

# Fixing the runtime column
dirty = '142 mins.'
number, text = dirty.split(' ')
clean = int(number)
print number

clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
data.head()

# Splitting up the genres
# determine the unique genres
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)

# make a column for each genre
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

# Removing year from title
data['title'] = [t[0:-7] for t in data.title]
data.head()

############################
# Explore Global Properties 
############################

data[['score', 'runtime', 'year', 'votes']].describe()

#hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])
#probably best to flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan

data.runtime.describe()

########################
# Make Some Basic Plots
########################

# more movies in recent years, but not *very* recent movies (they haven't had time to receive lots of votes yet?)
plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
plt.xlabel("Release Year")

plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")

plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")

#hmm, more bad, recent movies. Real, or a selection bias?
plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")

plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')

#########################
# Identify Some Outliers
#########################

# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]
# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]
# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

##########################################
# Run aggregation functions over the data
##########################################

# sum sums over rows by default
genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})

#axis=1 sums over columns instead
genre_count = data[genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

###########################
# Explore Group Properties
###########################

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

# compute the scatter in each year 
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

# You can also iterate over a GroupBy object. Each iteration yields two variables: one of the distinct values of the group key, and the subset of the dataframe where the key equals that value. To find the most popular movie each year:
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values

###############
# Observations
###############

# It appears that Western movies have gotten worse over time but may be on the rise again (probably not enough data points)
data[(data.Western ==  True)].groupby(decade).score.mean()

# The 2000s had the most movies represented, while the 2010s had the fewest movies represented
data.groupby(decade).size()

# Average scores per genre per decade (EXTRA)
for col in data.columns[7:]:
    dmean = data[data[col] == True].groupby(decade).score.mean()
    print col
    plt.plot(dmean.index, dmean.values, 'o-', label=col)  
plt.legend(frameon=False)

# Mean votes per movie have increased over the decades
decade_vote = data.groupby(decade).votes.mean() 
data.groupby(decade).size()
plt.plot(decade_vote.index, decade_vote.values, 'o-',
        color='r', lw=3, label='Decade Votes')
plt.scatter(data.year, data.votes, alpha=.04, lw=0, color='k')
plt.title("Mean Votes Per Decade")
plt.xlabel("Year")
plt.ylabel("Votes")
plt.legend(frameon=False)
plt.savefig(os.getcwd() + "/mean_votes")

# Mean runtimes have stayed consistent throughout the decades
decade_runtime = data.groupby(decade).runtime.mean()
decade_runtime.name = 'Decade Runtime Mean'
plt.plot(decade_runtime.index, decade_runtime.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.runtime, alpha=.04, lw=0, color='k')
plt.title("Mean Runtimes Per Decade")
plt.xlabel("Year")
plt.ylabel("Runtime")
plt.legend(frameon=False)
plt.savefig(os.getcwd() + "/mean_runtime")
