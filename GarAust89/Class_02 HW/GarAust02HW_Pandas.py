# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:03:44 2015

@author: garauste
"""

%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#tell pandas to display wide tables as pretty HTML tables
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        

## Pull in Data from URL ## 

data_url = 'https://raw.githubusercontent.com/cs109/content/master/imdb_top_10000.txt'

names = ['imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres']
data = pd.read_csv(data_url, delimiter='\t', names=names).dropna()
print "Number of rows: %i" % data.shape[0]
data.head()  # print the first 5 rows

## Number Text Conversion - Removal of the mins ##
dirty = '142 mins.'
number, text = dirty.split(' ')
clean = int(number)
print number

## Package the above up into a list compreshension to run over all variables ##
clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime
data.head()

#determine the unique genres
genres = set()
for m in data.genres:
    genres.update(g for g in m.split('|')) ##update function ? 
genres = sorted(genres)

#make a column for each genre
for genre in genres:
    data[genre] = [genre in movie.split('|') for movie in data.genres]
         
data.head()

## Removing the year from the title by stripping off last 7 characters ##
data['title'] = [t[0:-7] for t in data.title] ## List Comprehension? 
data.head()

## Describe is used to return some summary statistics from the dataset ##
data[['score', 'runtime', 'year', 'votes']].describe()

#hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])

#probably best to flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan ## What is np.nan 

## Using describe to summarize again after flagging bad variables ##
data.runtime.describe()

########################################
######### Make some Basic Plots ########
########################################

# more movies in recent years, but not *very* recent movies (they haven't had time to receive lots of votes yet?)

## Hist of years
plt.hist(data.year, bins=np.arange(1950, 2013), color='#cccccc')
plt.xlabel("Release Year")
remove_border()

## Hist of IMDB Ratings ##
plt.hist(data.score, bins=20, color='#cccccc')
plt.xlabel("IMDB rating")
remove_border()

## Hist of Runtimes ##
plt.hist(data.runtime.dropna(), bins=50, color='#cccccc')
plt.xlabel("Runtime distribution")
remove_border()

#hmm, more bad, recent movies. Real, or a selection bias?

plt.scatter(data.year, data.score, lw=0, alpha=.08, color='k')
plt.xlabel("Year")
plt.ylabel("IMDB Rating")
remove_border()

plt.scatter(data.votes, data.score, lw=0, alpha=.2, color='k')
plt.xlabel("Number of Votes")
plt.ylabel("IMDB Rating")
plt.xscale('log')
remove_border()

# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]

# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

#sum sums over rows by default
genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})

#axis=1 sums over columns instead
genre_count = data[genres].sum(axis=1) 
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

## Splitting up the movies by decade
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
remove_border()

## Computing the Scatter in each year also

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
remove_border()

## To find the most popular movie in each year
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values

######### Splitting up the movies by Genre and plotting ########
    
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
    remove_border(ax, left=False)
    ax.set_xlabel('Year')
    
######### Further plotting of genres ########
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8), tight_layout=True)

bins = np.arange(30, 240, 10)

for ax, genre in zip(axes.ravel(), genres):
    ax.hist(data[data[genre] == 1].runtime, 
            bins=bins, histtype='stepfilled', color='r', ec='none', alpha=.3, normed=True)
               
    ax.hist(data.runtime, bins=bins, normed=True,
            histtype='stepfilled', ec='none', color='#cccccc',
            zorder=0)
    
    ax.set_xticks(np.arange(30, 240, 60))
    ax.set_yticks([])
    ax.set_xlabel("Runtime [min]")
    remove_border(ax, left=False)
    ax.annotate(genre, xy=(230, .02), ha='right', fontsize=12)
    


############ And More Plots ########
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(12, 8), tight_layout=True)

bins = np.arange(0, 10, .5)

for ax, genre in zip(axes.ravel(), genres):
    ax.hist(data[data[genre] == 1].score, 
            bins=bins, histtype='stepfilled', color='r', ec='none', alpha=.3, normed=True)
               
    ax.hist(data.score, bins=bins, normed=True,
            histtype='stepfilled', ec='none', color='#cccccc',
            zorder=0)
    
    ax.set_yticks([])
    ax.set_xlabel("Score")
    remove_border(ax, left=False)
    ax.set_ylim(0, .4)
    ax.annotate(genre, xy=(0, .2), ha='left', fontsize=12)
    

##############################################
########### Gareth EDA Analysis ##############
##############################################

data

## Consider scatter plot of runtime and rating to check for correlation ##
cor_matrix = np.corrcoef(data.ix[:,2:6])
print cor_matrix 

plt.scatter(data.runtime,data.score,alpha=0.4,lw=0)
plt.xlabel('Runtime')
plt.ylabel('Score')

## Find the movies with runtime over 400 minutes ##
long_movies = data.title[data.runtime>400]
print long_movies

## Removing outlying runtime films ##
plt.scatter(data.runtime[data.runtime<400],data.score[data.runtime<400],alpha=0.4,lw=0)
plt.xlabel('Runtime')
plt.ylabel('Score')


## Create a 3D Scatter plot of year, runtime and rating
from mpl_toolkits.mplot3d import Axes3D 
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
x = data.year
y = data.runtime
z = data.score
 
ax.scatter(x, y, z, c='r', marker='o')
 
ax.set_xlabel('Year')
ax.set_ylabel('Runtime')
ax.set_zlabel('Score')
 
plt.show()

## This plot allows us to examine the change in runtime and score over the years. There have been significantly more film releases since the 1980s and we can also see some clustering of films around the 100-200 minute mark. This chart is not particularly useful as it is overcrowded. In the next chart we will examine the same variables but limit the runtime to between 100 and 200 minutes

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
x = data.year[data.runtime<200]
y = data.runtime[data.runtime<200]
z = data.score[data.runtime<200]
 
ax.scatter(x, y, z, c='r', marker='o')
 
ax.set_xlabel('Year')
ax.set_ylabel('Runtime')
ax.set_zlabel('Score')
 
plt.show()

### Creating a 3D Scatter of films with a runtime over 150 minutes 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
x = data.year[data.runtime>150]
y = data.runtime[data.runtime>150]
z = data.score[data.runtime>150]
 
ax.scatter(x, y, z, c='r', marker='o')
 
ax.set_xlabel('Year')
ax.set_ylabel('Runtime')
ax.set_zlabel('Score')
 
plt.show() 

## finding the average score of movies with run time over 150 minutes and over 200 minutes

data.score[data.runtime>100].sum()/data.score[data.runtime>100].count()
data.score[data.runtime>150].sum()/data.score[data.runtime>150].count()
data.score[data.runtime>200].sum()/data.score[data.runtime>200].count()

<<<<<<< HEAD:GarAust89/Class_02 HW/GarAust02HW_Pandas.py
## From the output of the three above functions we can see that the rating of movies increases with run time
=======
## From the output of the three above functions we can see that the rating of movies increases with run time
## However there are not as many longer movies and therefore we have a smaller sample size and cannot rely upon this result
>>>>>>> origin/master:GarAust89/GarAust02HW_Pandas.py
