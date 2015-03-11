# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 15:40:44 2015

@author: jchen
"""

# import packages
import sqlite3 
import pandas as pd
import numpy as np

# connect to the baseball database
conn = sqlite3.connect('/Users/jchen/Documents/SQLite/lahman2013.sqlite')

# create an object that holds the SQL query.
# Let's look at the average WHIP by left-handed v. right-handed pitchers over time
# Pull data on pitching stats
sql = """
select p.playerID,
	m.nameGiven as player_name,
	m.throws as throwing_hand,
	p.yearID as year,
	p.teamID,
	t.name as team_name,
	p.BB as walks,
	p.H as hits,
	p.IPOuts/3.0 as innings_pitched
from Pitching p
left join Master m on p.playerID=m.playerID
left join Teams t on p.teamID=t.teamID and p.yearID=t.yearID
group by 1,2,3,4,5,6,7,8,9
"""

# pass the connection and the SQL string to pandas
df = pd.read_sql(sql, conn)

# create new column: walks plus hits per inning pitched (WHIP)
# WHIP = (walks + hits)/(innings pitched)
df['whip']=(df.walks+df.hits)/df.innings_pitched

# create decade column
df['decade']=df.year//10 * 10

# describe the data
df.describe()

# Looks like we have some infinite/NaN values
# turn inf into NaN and drop them
df_clean = df.replace(np.inf, np.nan).dropna(how="all")


# Calculate the average whip across players in a single season by decade and pitching hand
# and put results into a data frame
df_new=pd.DataFrame({'avg_whip' : df_clean.groupby(['decade','throwing_hand']).whip.mean()}) # That looks weird though
df_new.index # Turns out the groupby gives df_new multi-level index

# Let's fix this
df_new=pd.DataFrame({'avg_whip' : df_clean.groupby(['decade','throwing_hand']).whip.mean()}).reset_index()

# write the summarized data back to SQLite
df_new.to_sql('avg_whip_per_decade_hand', conn, if_exists = 'replace')

# close the connection
conn.close()
