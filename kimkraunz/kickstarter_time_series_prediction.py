# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:01:26 2015

@author: frontlines
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import pandas
from pandas import DatetimeIndex
import numpy as np
from pandas.tseries.tools import to_datetime
import numpy
import matplotlib.pylab as plt
from __future__ import division

df = pandas.read_csv('/Users/frontlines/Documents/ds.csv')

pandas.set_option('display.max_columns', None)
df.head()
df.drop('currency_symbol', axis = 1, inplace = True)

# Data manipulation

# turn upper cases into lower cases in category and sub_category
df.main_category = df.main_category.str.lower()
df.sub_category = df.sub_category.str.lower()

art = ('conceptual_art', 'digital_art', 'illustration', 'installations', 'mixed_media', 'painting', 'performance_art', 'public_art', 'sculpture', 'textiles', 'video_art', 'ceramics')

comics = ('anthologies', 'comic_books', 'events', 'graphic_novels', 'webcomics')

crafts = ('candles', 'crochet', 'diy', 'embroidery', 'glass', 'knitting', 'letterpress', 'pottery', 'printing', 'quilts', 'stationery', 'taxidermy', 'weaving', 'woodworking')

dance = ('performances', 'residencies', 'spaces', 'workshops')

design = ('architecture', 'civic_design', 'graphic_design', 'interactive_design', 'product_design', 'typography')

fashion = ('accessories', 'apparel', 'childrenswear', 'couture', 'footwear', 'jewelry', 'pet_fashion', 'ready-to-wear')

film_and_video = ('action', 'animation', 'comedy', 'documentary', 'drama', 'experimental', 'family', 'fantasy', 'festivals', 'horror', 'movie_theaters', 'music_videos', 'narrative_film', 'romance', 'science_fiction', 'shorts', 'television', 'thrillers', 'webseries')

food = ('bacon', 'community_gardens', 'cookbooks', 'drinks', 'events', 'farmers_markets', 'farms', 'food_trucks', 'restaurants', 'small_batch', 'spaces', 'vegan')

games = ('gaming_hardware', 'live_games', 'mobile_games', 'playing_cards', 'puzzles', 'tabletop_games', 'video_games')

journalism = ('audio', 'photo', 'prints', 'video', 'web')

music = ('blues', 'chiptune', 'classical_music', 'country_&_folk', 'electronic_music', 'faith', 'hip-hop', 'indie_rock', 'jazz', 'kids', 'latin', 'metal', 'pop', 'punk', 'r&b', 'rock', 'world_music')

animals = ('fine_art', 'nature', 'people', 'photobooks', 'places')

technology = ('3d_printing', 'apps','camera_equipment', 'diy_electronics', 'fabrication_tools', 'flight', 'gadgets', 'hardware', 'makerspaces', 'robots','software','sound', 'space_exploration', 'wearables', 'web', 'open_software')

theater = ('experimental', 'festivals', 'immersive', 'musical', 'plays', 'spaces')

publishing = ('academic', 'anthologies', 'art_books','calendars','childrens_books','fiction', 'literary_journals', 'nonfiction', 'periodicals', 'poetry', 'radio_and_podcasts', 'translations', 'young_adult', 'zines')

df.main_category[df.main_category == "children's_book"] = 'childrens_books'
df.main_category[df.main_category == "children's_books"] = 'childrens_books'
df.main_category[df.main_category == 'short_film'] = 'shorts'
df.main_category[df.main_category == 'art_book'] = 'art_books'
df.main_category[df.main_category == 'periodical'] = 'periodicals'
df.main_category[df.main_category == 'radio_&_podcast'] = 'radio_and_podcasts'
df.main_category[df.main_category == 'radio_&_podcasts'] = 'radio_and_podcasts'
df.main_category[df.main_category == "farmer's_markets"] = 'farmers_markets'
df.main_category[df.main_category == 'print'] = 'prints'
df.main_category[df.main_category == 'film_&_video'] = 'film_and_video'


for name in technology:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'technology'

for name in theater:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'theater'

for name in animals:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'animals'

for name in music:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'music'
    
for name in journalism:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'journalism'
 
for name in games:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'games' 
    
for name in food:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'food' 

for name in film_and_video:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'film_and_video'
    
for name in fashion:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'fashion' 
    
for name in design:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'design'
    
for name in dance:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'dance' 
    
for name in crafts:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'crafts' 
    
for name in comics:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'comics' 
    
for name in art:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'art' 
    
for name in publishing:
    df.sub_category[df.main_category == name] = name
    df.main_category[df.sub_category == name] = 'publishing' 
     
df.main_category.value_counts(normalize = True, dropna = False )

main_cats = ('film_and_video', 'music','publishing', 'games', 'design', 'art','food','technology','fashion','comics', 'theater', 'crafts', 'journalism','photography','animals','dance')

df.sub_category[df.main_category == 'unknown'] = 'unknown'
df.sub_category[df.sub_category == 'film_&_video'] = 'film_and_video'

for name in main_cats:
    df.sub_category[df.sub_category == name] = 'unknown'
    
df.sub_category.value_counts(normalize = True, dropna = False)

df['funded']= 2
df.funded[df.state == 'successful'] = 1
df.funded[df.state == 'failed'] = 0
df = df[df.funded != 2]

df.deadline = to_datetime(df.deadline)
df['year'] = DatetimeIndex(df['deadline']).year
df['month'] = DatetimeIndex(df['deadline']).month


# Convert pledged to USD

df['pledged_USD'] = df.pledged

df.pledged_USD = df.pledged_USD[df.currency == "GBP"] = df.pledged * 1.48
df.pledged_USD = df.pledged_USD[df.currency == "CAD"] = df.pledged * .79
df.pledged_USD = df.pledged_USD[df.currency == "AUD"] = df.pledged * .76
df.pledged_USD = df.pledged_USD[df.currency == "EUR"] = df.pledged * 1.07
df.pledged_USD = df.pledged_USD[df.currency == "NZD"] = df.pledged * .75
df.pledged_USD = df.pledged_USD[df.currency == "DKK"] = df.pledged * .14
df.pledged_USD = df.pledged_USD[df.currency == "NOK"] = df.pledged * .12

df.pledged_USD = df.pledged_USD.astype('int')


# Set up dataframe for each of the main_categories

df_art = df.set_index(df.deadline)
df_art[['deadline', 'pledged_USD']].sort('deadline').resample('Q', how = 'sum')



''' not working.  getting the following error:
IndexingError: Too many indexers'''

for i in range(1,(len(df_art) - 4)):
    y_art = df_art.goal_USD.ix[i:(i + 4), :].T
    
# make sure it's a dataframe and give it column names
X = 
y = 

# Run linear regression with cross validation  


lr_model = LinearRegression(fit_intercept = True)
  
 
accuracy_scores = cross_val_score(lr_model, X, y, cv=10, scoring = 'accuracy', verbose = True)

lr_model.fit(X, y)

print accuracy_scores


# Predict future values and fill dataframe with predictions

y_predictions = lr_model.predict(y_new)

