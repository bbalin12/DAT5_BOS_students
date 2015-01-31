# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#***Class Two Homework***

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#building the data frame
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv('http://bit.ly/cs109_imdb', delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head() 



#cleaning the data frame

#fixing runtime column to make numeric
dirty = '142 mins.'
number, text = dirty.split(' ')
clean = int(number)
print number

clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
data.head()

#determine the unique genres
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)

#make a column for each genre
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

#remove year from titile
data['title'] = [t[0:-7] for t in data.title]
data.head()



#explore global properties
data[['score', 'runtime', 'year', 'votes']].describe()
#hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])
#probably best to flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan
data.runtime.describe()

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

#identifying some outliers
# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]
# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]
# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]


#run aggregation functions like sum over several rows or columns
#what genres are the most frequent?
#sum sums over rows by default
genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})

#how many genres does a movie have, on average?
#axis=1 sums over columns instead
genre_count = data[genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()
#average movie has 2.75 genres


#explore group properties
#let's split up movies by decade 
decade=(data.year //10)*10
tyd = data[['title', 'year']]
tyd['decade'] = decade
tyd.head()

#GroupBy will gather movies into groups with equal decade value
#mean score for all movies in each decade
decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

plt.plot(decade_mean.index, decade_mean.values, 'o-', color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

#we can go one further, and compute the scatter in each year as well
grouped_scores = data.groupby(decade).score

mean = grouped_scores.mean()
std = grouped_scores.std()

plt.plot(decade_mean.index, decade_mean.values, 'o-', color='r', lw=3, label='Decade Average')
plt.fill_between(decade_mean.index, (decade_mean + std).values, (decade_mean - std).values, color='r', alpha=.2)
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

#You can also iterate over a GroupBy object. Each iteration yields two variables: 
#one of the distinct values of the group key, and the subset of the dataframe where the key equals that value. 
#To find the most popular movie each year:
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values


#finding two interesting facts:
data.describe()
data.runtime.describe()
data['long_runtime']=0
data.long_runtime[data.runtime>115]=1 #set long_runtime when runtime is over 115
data.score[data.long_runtime==1].mean()
data.score[data.long_runtime==1].max()
data.score[data.long_runtime==1].min()
#For movies that have a long runtime (>115), the average score for those movies is 6.99.
#The maximum score for the movies that have long run times is 9.2 whereas the min is 1.9

data[data.score==data.score.max()].year
#the movies that have the higest scores were in 1994 and 1972
#let's see what movies they were and other info:
data[(data.score > 9)][['title', 'year', 'score', 'votes', 'genres', 'long_runtime']]
#"The Shawshank Redpemtion" and "The Godfather" were the highest scored movies (9.2 score) which were
    #both crime and drama films

#finding two plots:
decade=(data.year //10)*10
tyd = data[['title', 'year','score', 'votes', 'runtime']]
tyd['decade'] = decade
tyd.groupby('decade').runtime.mean().plot(kind='bar', title='Runtimes per Year', ylim=(80,120))
plt.xlabel("Year")
plt.ylabel("Runtime")
#plot of mean runtimes for each decade

plt.hist(data.year.dropna(), bins=20, color='#cccccc')
plt.xlabel("Year distribution")
#histogram of years in data set




































