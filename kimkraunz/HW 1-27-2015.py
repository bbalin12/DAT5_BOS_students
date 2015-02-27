# -*- coding: utf-8 -*-


import pandas


# gives column names
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
# assigns the input data to data using the url, creates columns based on tab deliminator, not sure of names=names, and drops any rows with missing data but doesn't change dataset
data = pandas.read_csv('https://raw.githubusercontent.com/cs109/content/master/imdb_top_10000.txt', delimiter='\t', names=names).dropna()
# prints number of rows with an integer, not sure of shape
print "Number of rows: %i" % data.shape[0]
data.head()  # print the first 5 rows

# creates a variable clean_runtime that takes the first part that is split and turns it into a float for each row in the column runtime
clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
# revalues each row in the column runtime with the new variable clean_runtime
data['runtime'] = clean_runtime
# prints out the first 5 rows
data.head()

#determine the unique genres
# creates an empty variable genres that will contain set of unique unordered items
genres = set()
# runs loop for which updates the new variable genres with unique genres **Do not fully understand**
for m in data.genres:
    genres.update(g for g in m.split('|'))
# sorts alphabetically the items in the new variable genres
genres = sorted(genres)
print genres


#make a column for each genre based on splitting at '|' ** do not fully understand**
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
# prints out first five rows
data.head()

# deletes the last 7 characters of the title in each row
data['title'] = [t[0:-7] for t in data.title]
# Can I write in a different for loop way?
data.head()

# outputs descriptive characteristics
data[['score', 'runtime', 'year', 'votes']].describe()

#hmmm, a runtime of 0 looks suspicious. How many movies have that?
# prints # of movies with runtime equal to 0.  ** don't understand use of len()**
print len(data[data.runtime == 0])

import numpy as np

#probably best to flag those bad data as NAN
# replaces 0 with nan in runtime
# why can you do this without loop and why np.nan
data.runtime[data.runtime==0] = np.nan

# outputs descriptive statistics for runtime
data.runtime.describe()

import matplotlib.pyplot as plt
# more movies in recent years, but not *very* recent movies (they haven't had time to receive lots of votes yet?)
# plots a histogram of years with bins in a range from 1950-2013 in light gray
plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
# gives x axis label of release year
plt.xlabel("Release Year")

# plots histogram of scores with 20 bins in light gray
plt.hist(data.score, bins=20, color = '#cccccc')
# gives x axis label of IMDB rating
plt.xlabel("IMDB rating")

# plots a histogram of runtime with 50 bins in light gray
plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
# gives x axis label of runtime distribution
plt.xlabel("Runtime distribution")

#hmm, more bad, recent movies. Real, or a selection bias?
# plots scatter plot of year on x axis and score on y axis
# ** not sure what lw=0, alpha=.08 or color='k' means
plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")


plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')

# low-score movies with lots of votes
# outputs the title, year, score, votes, and genres of any movies with votes >9e4 and scores <5
data[['title', 'year', 'score', 'votes', 'genres']][(data.votes > 9e4) & (data.score < 5)]

# The lowest rated movies
# outputs the movie with the lowest score
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest rated movies
# outputs the movie with the highst score
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

import numpy as np
#sum sums over rows by default
# ** do not understand**
genre_count = np.sort(data[genres].sum())[::-1]
pandas.DataFrame({'Genre Count': genre_count})

#axis=1 sums over columns instead
# creates new variable genre_count that is equal to the sums of data[genres] in columns
# don't understand if data[genres] is now the set of genres
# uses %0.2f to signiify a float with 2 decimals
# also uses a function built into a variable in genre_count.mean()
genre_count = data[genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

# removes last digit an then multiplies by ten to get decade 
decade =  (data.year // 10) * 10
decade.head()
data.head()

# creates new variable or list? tyd bu I don't see it when I print out data
# is that because it is separate from data?
tyd = data[['title', 'year']]
data.head()
print tyd
tyd['decade'] = decade

tyd.head()

#mean score for all movies in each decade
# creates variable decade_mean that is a list? of grouped by decade and taken the mean of the variable  score
decade_mean = data.groupby(decade).score.mean()
decade_mean.name = 'Decade Mean'
print decade_mean

# creates x y plot with year (index) and mean (values)
# not sure about decade_mean.index and decade_mean.values function
plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
# creates scatter with year vs score
# not sure about alpha, lw, or color = 'k'
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
# shows legend but without border
plt.legend(frameon=False)

# creates new variable grouped_scores based on group by and score
# *** not sure here if creating list
grouped_scores = data.groupby(decade).score

# finds the mean and std of the grouped_scores
mean = grouped_scores.mean()
std = grouped_scores.std()

# creates x y plot of year(index) and average(value)
plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
# creates red filling between decade +- std
plt.fill_between(decade_mean.index, (decade_mean + std).values,
                 (decade_mean - std).values, color='r', alpha=.2)
# creates scatter plot of year vs score
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

# prints out the highest core by year

for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values
    
# histogram of votes with log scale 
plt.hist(data.votes.dropna(), bins=1000, color='#cccccc')
plt.xlabel("Votes")
plt.xscale('log')
plt.ylabel("# of Movies")

# xy scatter of runtime vs score
plt.scatter(data.runtime, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Runtime")
plt.ylabel("IMDB Rating")


# xy scatter of year vs votes
plt.scatter(data.year, data.votes, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("Votes")

# figure out how to do by decade!!!
data.plot(kind='scatter'(by = decade), x='votes', y='score')

# The lowest voted movie
# outputs the movie with the lowest votes
data[data.votes == data.votes.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest voted movie
# outputs the movie with the most votes
data[data.votes == data.votes.max()][['title', 'year', 'score', 'votes', 'genres']]

data[data.runtime == data.runtime.max()][['title', 'year', 'runtime', 'score', 'votes', 'genres']]

data[data.runtime >= 180].describe()
data[data.runtime < 180].describe()

data.sort_index(by = 'runtime').dropna().tail()

data[(data.score >= 5) & (data.runtime < 120) & (data.votes > 100000) & (data.Romance == 1) & (data.Comedy == 1)]



'''
Shawshank Redemption received the most votes at 619,479(as well as the highest score- see above).  
Garage, Mimino, The Quiller Memorandum, Taal, and The Navigators received the least number of 
votes at 1,356.  Santantango, a movie released in 1994 is the longest, at 450 minutes.  There 
are 4 movies that have a runtime longer than four hours.  There are 90 movies that have a 
runtime equal or longer than 3 hours.  The average score for movies with a runtime less than 
180 minutes is 6.37 while for those with a runtime greater than or equal to 180 minutes it's 7.61.  
The median release year for movies with a runtime less than 180 minutes is 1998 while the
median release year for movies with a runtime greater than or equal to 180 minutes is 1991.

    






