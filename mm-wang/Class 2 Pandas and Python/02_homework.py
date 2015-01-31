# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 21:31:50 2015

@author: Margaret
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

names = ['imdbID','title','year','score','votes','runtime','genres']
data = pd.read_csv('https://raw.githubusercontent.com/cs109/content/master/imdb_top_10000.txt',delimiter='\t',names=names).dropna()

print "Number of rows: %d" % data.shape[0]
#data.head()

#fix the runtime frame to just display a number
clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
#data.head()

#determine the unique genres of movies
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|')) #updates genres with each element in list m once split
genres = sorted(genres)

#make a column for each genre
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres] 
    #for each movie, if genre is in the split list, change the value to True

#data.head()

data['title']= [t[0:-7] for t in data.title] #for each title in data.title, take the title up to -7 characters
#data.head()

data[['score','runtime','year','votes']].describe()

#how many movies have a runtime of 0?
print len(data[data.runtime==0])

#flag those movies as NaN for bad data
data.runtime[data.runtime==0] = np.nan

#how many movies have a runtime of 0?
print len(data[data.runtime==0])

#flag those movies as NaN for bad data
data.runtime[data.runtime==0] = np.nan

plt.figure(0)
plt.hist(data.year,bins=np.arange(1950,2013),color='#cccccc')
plt.xlabel("Release Year")

plt.figure(1)
plt.hist(data.score,bins=20,color='#cccccc')
plt.xlabel("IMDB rating")

plt.figure(2)
plt.hist(data.runtime.dropna(),bins=50,color='#cccccc')
plt.xlabel("Runtime Distribution")

#determining if the bad, recent movies is due to selection bias
plt.figure(3)
plt.scatter(data.year, data.score,lw=0,alpha=0.08,color='c')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")

#votes and rating
plt.figure(4)
plt.scatter(data.votes,data.score,lw=0,alpha=0.2,color='y')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')

#find low score movies with a lot of votes
data[(data.votes>9e4) & (data.score<5)][['title','year','score','votes','genres']]

#lowest and highest rated movies
data[data.score == data.score.min()][['title','year','score','votes','genres']]
data[data.score == data.score.max()][['title','year','score','votes','genres']]

#sum sums over rows
#why doesn't it display the genres? because I had to add index=genres after the data frame!
genre_count = np.sort(data[genres].sum())[::-1]
print pd.DataFrame({'Genre Count': genre_count})

#axis=1 sums over columns
genre_count = data[genres].sum(axis=1)
print "Average movie has %0.2f genres" %genre_count.mean()
genre_count.describe()

##Group Properties
#splitting movies up by decade

decade = (data.year // 10) * 10

tyd = data[['title','year']]
tyd['decade'] = decade
#tyd.head()

#mean scores in each deccade

decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

plt.figure(5)
plt.plot(decade_mean.index,decade_mean.values,'o-',color='b',lw=3,label='Decade Average')
plt.scatter(data.year,data.score,alpha=0.04,lw=0,color='g')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

grouped_scores = data.groupby(decade).score

mean = grouped_scores.mean()
std = grouped_scores.std()

plt.figure(6)
plt.plot(decade_mean.index,decade_mean.values,'o-',color='m',lw=3,label = "Decade Average")
plt.fill_between(decade_mean.index,(decade_mean+std).values,(decade_mean-std).values,color='m',alpha=0.2)
plt.scatter(data.year,data.score,alpha=0.04,lw=0,color='c')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

#group movies by release year
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values


### HOMEWORK

# Interesting Fact #1: The decades with the highest score are 1970 and 1990
# Max Score per Decade
decade_max = data.groupby(decade).score.max()
decade_max.name = 'Decade Max'
print decade_max


# Interesting Fact #2: The average votes on each movie increase over time
tvyd = data[['title','votes','year']]
tvyd['decade']=decade
tvyd.head()

votes_decademean = data.groupby(decade).votes.mean()
votes_decademean.name = "Mean Votes By Decade"

print votes_decademean


# Plot #1
# Plots the number of votes in each decade
plt.figure(7)
plt.scatter(tvyd.decade,data.votes,lw=0,alpha=0.4,color='r')
plt.xlabel("Decade")
plt.ylabel("Number of Votes")

# Plot #2
# Plots the ratings in each decade
plt.figure(8)
plt.scatter(tvyd.decade,data.score,lw=0,alpha=0.4,color='g')
plt.xlabel("Decade")
plt.ylabel("IMDB Rating")

