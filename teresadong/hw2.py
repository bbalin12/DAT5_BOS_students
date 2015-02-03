# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 23:21:10 2015

@author: tdong1
"""

#just include import statments from the first block
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#specify data url
data_url ='https://raw.githubusercontent.com/cs109/content/master/imdb_top_10000.txt'

#Build a Data Frame
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv(data_url, delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head()  # print the first 5 rows

#Clean the Data Frame
#Fix the runtime column
dirty = '142 mins.'
number, text = dirty.split(' ')
clean = int(number)
print number

#Use list comprehension to clean the entire runtime column
clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
data.head()

#Split the single drama column to dummy boolean variables
#determine the unique genres
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|'))
genres = sorted(genres)

#make a column for each genre
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

#Remove year from the title column
data['title'] = [t[0:-7] for t in data.title]
data.head()

#Explore global properties
#calling describe on relevant columns
data[['score', 'runtime', 'year', 'votes']].describe()

#set movies with no runtime to NA
#hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])

#probably best to flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan

#now check the runtime column again
data.runtime.describe()
#we see that the count has changed as NaN are not counted

#make basic plots
#look at count of movies by release year
# more movies in recent years, but not *very* recent movies (they haven't had time to receive lots of votes yet?)
plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
plt.xlabel("Release Year")
#remove_border()

#now look at distribution of movies by IMDB random
plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")
#remove_border()

#now look at distribution of movies by their runtime
plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")
#remove_border()

#hmm, more bad, recent movies. Real, or a selection bias?
plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")
#remove_border()


#now look at scattterplot of number of votes by IMDB rating
plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')
#remove_border()

#identify some outliers
# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]

# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

#Run aggregation functions like sum over several rows or columns
#What genres are the most frequent?
#sum sums over rows by default
genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})

#For some reason first column disappeared
#use the pandas sorting function instead
print data[genres].sum().order(ascending=False)

#How many genres does a movie have, on average?
#axis=1 sums over columns instead
genre_count = data[genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

#Explore Group Properties
#Let's split up movies by decade
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
#remove_border()

#Add in movies within one standard deviation
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
#remove_border()

#find most popular video each year
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values
    
#Ignoring Small Multiples section

#Now for own work
#Create two new plots that show something interesting
print data.head()

#Is there a relationship between runtime and rating?
plt.scatter(data.runtime, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Runtime")
plt.ylabel("IMDB Rating")
#There appears to be a weak positive relationship between runtime and rating

#Has the length of videos increased as years gone on
plt.scatter(data.year, data.runtime, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("Runtime")
#Looks like movies have gotten longer as years increased


for year, subset in data.groupby('Action'):
    print year, subset[subset.score == subset.score.max()].title.values
#The most popular action movies are Inception and the Dark Knight

for year, subset in data.groupby('Adult'):
    print year, subset[subset.score == subset.score.max()].title.values    
 #The most popular adult movie is Import/Export   
    





