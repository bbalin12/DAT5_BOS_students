'''
jonblum
2015-01-22
datbos05
class 2 hw
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load data and create column headings
names = ['imdbID','title','year','score','votes','runtime','genres']
data = pd.read_csv('https://raw.githubusercontent.com/cs109/content/master/imdb_top_10000.txt', delimiter='\t', names=names).dropna()

# get info
print 'Number of Rows: %i' %  data.shape[0] # why not len() or .count?
data.head() # first 5 rows
data.describe()


# make runtime numerical
clean_runtime = [float(r.split(' ')[0]) for r in data.runtime]
data['runtime'] = clean_runtime

# splitting up genre lists into boolean indicator variables

##  determine the unique genres
genres = set()
for m in data.genres:
	genres.update(g for g in m.split('|'))
genres = sorted(genres)

## make a column for each genre
for genre in genres:
	data[genre] = [genre in movie.split('|') for movie in data.genres]

# strip year from title
data['title'] = [t[0:-7] for t in data.title]

data.head(10)


data[['score', 'runtime', 'year', 'votes']].describe()

# hmmm, a runtime of 0 looks suspicious. How many movies have that?
print len(data[data.runtime == 0])

# probably best to flag those bad data as NAN
data.runtime[data.runtime==0] = np.nan

# that's better:
data.runtime.describe()

# more movies in recent years, but not *very* recent movies (they haven't had time to receive lots of votes yet?)
year_hist = data.year.hist(bins=np.arange(1950, 2013), color='#cccccc')
year_hist.set_xlabel("Release Year")

# ratings histogram
rating_hist = data.score.hist(bins=20, color='#cccccc')
rating_hist.set_xlabel("IMDB rating")

# runtime histogram
runtime_hist = data.runtime.dropna().hist(bins=50, color='#cccccc')
runtime_hist.set_xlabel("Runtime distribution")

#hmm, more bad, recent movies. Real, or a selection bias?

year_score_scatter = data.plot(kind='scatter', x='year', y='score', lw=0, alpha=.08, color='k')
year_score_scatter.set_xlabel("Year")
year_score_scatter.set_ylabel("IMDB Rating")


vote_score_scatter = data.plot(kind='scatter', x='votes', y='score', lw=0, alpha=.2, color='k')
vote_score_scatter.set_xlabel("Number of Votes")
vote_score_scatter.set_ylabel("IMDB Rating")
vote_score_scatter.set_xscale('log')

# low-score movies with lots of votes
data[(data.votes > 9e4) & (data.score < 5)][['title', 'year', 'score', 'votes', 'genres']]

# The lowest rated movies
data[data.score == data.score.min()][['title', 'year', 'score', 'votes', 'genres']]

# The highest rated movies
data[data.score == data.score.max()][['title', 'year', 'score', 'votes', 'genres']]

# What genres are the most frequent?
# sum sums over rows by default
genre_count = np.sort(data[genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})

# how many genres does a movie have on average?
# axis=1 sums over columns instead
genre_count = data[genres].sum(axis=1)
print "Average movie has %0.2f genres" % genre_count.mean()
genre_count.describe()

# which movies hav 8 genres?
data[data[genres].sum(axis=1) == max(data[genres].sum(axis=1))]

# splitting up movies by decade
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

# iterate over groupby to find best movie of the year:
for year, subset in data.groupby('year'):
    print year, subset[subset.score == subset.score.max()].title.values




# 1. Check relationship between movie length and rating for normal lengths:

runtime_score_scatter = data[(data.runtime < 240) & (data.runtime > 60)].plot(kind='scatter', x='runtime', y='score', lw=0, alpha=.08, color='k')
runtime_score_scatter.set_xlabel("Runtime (min)")
runtime_score_scatter.set_ylabel("IMDB Rating")

# There does tend to be a correlation between runtime and score -- particularly at the extremes.
# Though there are plenty of highly-rated short films, longer-films do tend to be highly-rated,
# and lower-rated films tend not be over two hours

# 2.  Genre representation in Top 500 Rated Films

# Which genres are over/under-represented in highest-rated movies?
top_movie_genres = data.sort_index(by='score').tail(500)[genres]

top_movie_genres_percent = top_movie_genres.sum()/top_movie_genres.sum().sum()
# top_movie_genres_percent.plot(kind='bar', title='Genre Representation in Top 500 Films')

# contrast with overall representation
all_movie_genres_percent = data[genres].sum()/data[genres].sum().sum()

# all_movie_genres_percent.plot(kind='bar', title='Genre Representation in ALL Films')

delta = (top_movie_genres_percent - all_movie_genres_percent) * 100

delta.plot(kind='bar', title='Percentage Over/Under-Represention of Genres in Top 500 Films')
# Drama is far and away the most rpresented genre.
# Comedy is under-represented