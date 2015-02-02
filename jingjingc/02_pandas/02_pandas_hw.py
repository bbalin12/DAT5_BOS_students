# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:41:01 2015

@author: jchen
"""

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Read in data from URL and put into data frame
# First create object with column names
names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv('https://raw.githubusercontent.com/cs109/content/master/imdb_top_10000.txt',
                   delimiter='\t', names=names).dropna() 
print "Number of rows: %i" % data.shape[0]
data.head()  # print the first 5 rows


# Example of splitting string
dirty = '142 mins.'
number, text = dirty.split(' ') # split string at space character, assign parts
clean = int(number) # convert first part to int
print number


# Clean runtime data
clean_runtime = [float(r.split(' ')[0]) for r in data.runtime] # create new column of integer runtime minutes
data['runtime'] = clean_runtime # replace old runtime column with cleaned data
data.head() 


# Determine the unique genres
genres = set() # new set object
for m in data.genres: # iterate through each row of genre values
    genres.update(g for g in m.split('|')) # update genre object with new values
genres = sorted(genres) # sort set of genres


# Make a column for each genre
for genre in genres: # iterate through all possible genres
    data[genre] = [genre in movie.split('|') for movie in data.genres] # set genre column to boolean based on existence in genres column
         
        
# Remove year from title
data['title'] = [t[0:-7] for t in data.title] # remove last 7 characters from title


# Describe relevant columns
data[['score', 'runtime', 'year', 'votes']].describe()

# hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])

# probably best to flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan

data.runtime.describe()

#########################
# Plotting

# Plot movies by release year
plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc') # histogram w/1 year bins
plt.xlabel("Release Year")
# more movies in recent years, but not *very* recent movies (they haven't had time to receive lots of votes yet?)

# Histogram of movies by IMDB score
plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")

# Histogram of movies by runtime
plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")


# Plot year against rating
plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")
# hmm, more bad, recent movies. Real, or a selection bias?
# lot more data on recent movies

# Plot votes against rating
plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log') # set x axis to log scale
# seems that highly rated movies get a lot of votes


# Identify outliers
# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']] 

# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

# Count movies by genre
genre_count = np.sort(data[genres].sum())[::-1] # sum sums over rows by default, return from bottom
pd.DataFrame({'Genre Count': genre_count}) # turn results into data frame

# Average genres per movie
genre_count = data[genres].sum(axis=1) # axis=1 sums over columns instead
print "Average movie has %0.2f genres" % genre_count.mean() # print average to 2 decimal places


# Movies by decade
decade =  (data.year // 10) * 10 # calculate the decade - mod by 10, multiply by 10

tyd = data[['title', 'year']] 
tyd['decade'] = decade


# Find the mean score for all movies in each decade
decade_mean = data.groupby(decade).score.mean() 
decade_mean.name = 'Decade Mean'
print decade_mean

# Plot mean score by decade
plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k') # add all movie scores by year in background
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)

# group scores by decade
grouped_scores = data.groupby(decade).score

# calculate mean and std on each group
mean = grouped_scores.mean()
std = grouped_scores.std()

# Plot decade mean with range of std
plt.plot(decade_mean.index, decade_mean.values, 'o-',
        color='r', lw=3, label='Decade Average')
plt.fill_between(decade_mean.index, (decade_mean + std).values,
                 (decade_mean - std).values, color='r', alpha=.2)
plt.scatter(data.year, data.score, alpha=.04, lw=0, color='k')
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend(frameon=False)


# Find the most popular movie in each year
for year, subset in data.groupby('year'): # iterate through each year, movies in that year
    print year, subset[subset.score == subset.score.max()].title.values # print year, movie with max score in that year
    
    
#############################  
# Homework - exploring data

# Look at movie ratings by genre
# For each genre, find the average rating, average runtime, number of movies, the highest rated movies and their ratings
mean_scores=[]
max_scores=[]
highest_rated=[]
genre_count=[]
mean_runtime=[]
for genre in genres:
    subset=data[data[genre]==True]
    mean_scores.append(round(subset.score.mean(),1))
    max_scores.append(subset.score.max())
    highest_rated.append(subset[subset.score == subset.score.max()].title.values)
    genre_count.append(subset.title.count())
    mean_runtime.append(round(subset.runtime.mean(),1))

# Concatenate results into data frame
genre_scores=pd.DataFrame({'Genre':genres, 'Movie Count':genre_count, 'Highest Rated Movie':highest_rated, 'Highest Rating':max_scores, 'Genre Mean Rating':mean_scores, 'Genre Mean Runtime':mean_runtime})

# Looks like film-noir has the highest average movie rating
# and while Reality TV has the lowest average movie rating, there is only 1 movie in that genre
# Horror movies, on average, seem to rank lower
genre_scores.sort('Genre Mean Rating', ascending=False) # sort by average rating descending   

# Looks like Historical movies have the longest runtime on average, followed by war movies and biographies
# (This is not terribly surprising)
genre_scores.sort('Genre Mean Runtime') # sort by average rating descending   


# Look at average rating of Comedy movies over time
# appears to decrease while movies/year increases
# I wonder if we can look at this across genres?
movie_genre='Comedy' # pick a genre
years=[]
mean_score=[]
movie_count=[]
for year, subset in data[data[movie_genre]==True].groupby('year'):
    years.append(year)
    mean_score.append(round(subset.score.mean(),1))
    movie_count.append(subset.title.count())
    
# combine results in data frame  
genre_year=pd.DataFrame({'year':years,'mean_score':mean_score,'movie_count':movie_count})
    

##################
# Plots

    
# Plot average rating of Drama movies by year
genre_plot=genre_year.plot(kind='line', x='year', y='mean_score', alpha=0.5, ylim=[0,10])
genre_plot.set_title('Mean Comedy Rating')

# Plot number of genre movies by release year
genre_data=data[data[movie_genre]==True]
genre_data.year.hist(bins=np.arange(1950,2013))

# Would be interesting to overlay the histogram and average rating in one plot

# Boxplot of rating by decade
data_decade=data
data_decade['decade']=decade
    
data_decade.boxplot(column='score', by='decade')


# Plot runtime against year
# Is there a trend in the runtime of movies over time?
plt.scatter(data.year, data.runtime, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("Runtime")

