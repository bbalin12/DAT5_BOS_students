# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 20:56:59 2015

@author: Haruo
"""

# importing the package for SQLite alongside pandas.
import sqlite3
import pandas

# connect to the baseball database. Notice I am passing the full path
# to the SQLite file.
conn = sqlite3.connect('C:\Users\mizutani\Documents\SQLite\lahman2013.sqlite')

# creating an object contraining a string that has the SQL query. Notice that
# I am using triple quotes to allow my query to exist on multiple lines.
sql = """SELECT sq1.first_year, ROUND(AVG(b.AB),2) FROM Batting b
LEFT OUTER JOIN 
(SELECT  playerID, MIN(yearID) as first_year from Batting 
WHERE teamID = 'NYA' GROUP BY playerID) sq1
ON b.playerID = sq1.playerID
AND b.yearID = sq1.first_year
WHERE b.yearID > 1980 AND teamID = 'NYA';"""

# passing the connection and the SQL string to pandas.read_sql.
df = pandas.read_sql(sql, conn)

# Alternative command
# df = pandas.io.sql.read_sql_query(sql, conn)

# NOTE: I can use this syntax for SQLite, but for other flavors of SQL
# (MySQL, PostgreSQL, etc.) you will have to create a SQLAlchemy engine 
# as the connection.  A PostgresSQL example is below.
# Stack Overflow also has some nice examples of how to make this connection.

# closing the connection.
conn.close()

# filling NaNs
df.fillna(0, inplace = True)

# re-opening the connection to SQLite.
conn = sqlite3.connect('C:\Users\mizutani\Documents\SQLite\lahman2013.sqlite')
# writing the table back to the database.
# If the table already exists, I'm opting to replace it.  
df.to_sql('AveTable2ndYankees', conn, if_exists = 'replace')
# You can also append to the table if it exists 
# with the option if_exists = 'append.'

# closing the connection.
conn.close()


## PostgreSQL example.  -- use 'engine' instead of 'conn' above.
#from sqlalchemy import create_engine
#engine = create_engine('postgresql://uname:password@999.999.999.999:5432/test_dev')