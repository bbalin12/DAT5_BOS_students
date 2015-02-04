# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 17:29:08 2015

@author: garauste
"""

import pandas
import sqlite3

## Connect to the baseball database by passing it the full path
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

## Creating an object that contains a string that holds a SQL query. Triple quotes are used to allow the query to exits on multiple lines
sql = """select coalesce(b.nameFirst,"") || coalesce(b.nameLast,"") as PlayerName,c.year_inducted, c.Career_HRs, c.Games, c.At_Bats 
from master b left outer join 
(select max(d.playerID) as playerID, sum(d.HR) as Career_HRs, sum(d.G) as Games, sum(d.AB) as At_Bats, max(e.yearID) as year_inducted 
from batting d left outer join HallOfFame e
	on d.playerID = e.playerID
	group by d.playerID) c
	on b.playerID = c.playerID
	where year_inducted is not null
	order by year_inducted desc"""

## Passing the connection and SQL query to pandas.read_sql
df = pandas.read_sql(sql,conn)

# Close the Connection 
conn.close()

## Adding some new columns ##
df['Average_HR_Per_Game'] = df.Career_HRs/df.Games

# Inspect the new data #
df.head()

# Sort the data frame by average HRs per Game #
df = df.sort(columns='Average_HR_Per_Game',ascending=False)

## Inspect new data ##
df.head()

########### Write Data Back to Data Base ##################

# ReOpen Connections
conn = sqlite3.connect('C:\Users\garauste\Documents\SQLite\lahman2013.sqlite')

# Write table back #
df.to_sql('Hall_of_Famers_Average_HRs',conn, if_exists = 'replace')

# Close Connection Again #
conn.close()

