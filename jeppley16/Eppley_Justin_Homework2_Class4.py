# -*- coding: utf-8 -*-
"""
Created on Tue Feb 03 09:32:19 2015

@author: jeppley
"""

/*************************HOMEWORK 2: Class 4: SQL and Pandas***********************************************************************/


#QUESTION 1: Find the player with the most at-bats in a single season (716, rolliji01 in 2007)

select yearID, playerID, sum(AB) as atbat
from batting b
group by yearID, playerID
order by ab desc

#QUESTION 2: Find the name of the player with most at-bats in baseball history (rosepe01, 14053)

select playerID, sum(AB) as ab2
from batting
group by playerID
order by ab2 desc;

#QUESTION 3: Find average number of at_bats of players in rookie season (58.12)

select avg(AB) as avgab from
#joining larger dataset to rookie year
(
select b.*, sq1.minyear from Batting b
left outer join 
#selecting players' rookie year to create identifier
(select playerID, min(yearID) as minyear
from batting
group by playerID) sq1 on sq1.playerID = b.playerID and b.yearID = sq1.minyear)
where minyear is not null;

#QUESTION 4: Find the average number of at_bats of players in their final season for all players born after 1980 (88.94).

create table peace as
select * from
( select b.*, sq1.maxyear from Batting b
left outer join
(select playerID, max(yearID) as maxyear
from batting
group by playerID) sq1 on sq1.playerID = b.playerID and b.yearID = sq1.maxyear)
where maxyear is not null;

select avg(AB) as avgabs from
(select b.*, c.birthYear from peace b
left outer join Master c on b.playerID = c.PlayerID
where c.birthYear >1980);

#QUESTION 5: Find the average number of at_bats of Yankees players who began their second season at or after 1980(105.65).

select * from teams
where name like '%Yank%'
order by teamID

#Goal is to get dataset culled down to desired parameters step by step#

#a, b to get c 


select avg(d.ab) as season2
from


(select playerid, min(yearid) as firstyearyankee from batting where teamid = 'NYA' group by playerid) a #database 1: first year with Yankees

inner join

(select playerid, yearid from batting where teamid = 'NYA') b #database 2: any year with Yankees

on a.playerid = b.playerid and a.firstyearyankee != b.yearid group by b.playerid having second_year_NYA >= 1980) c #database 3, actual subsetting

inner join

(select playerid, ab from batting where teamid = 'NYA') d #final database you want to pull from

on c.playerid = d.playerid

#Novel Thing 1: Which team has the highest average weight of all time? (Answer: MIA, at 216 pounds)


select teamID, avg(weight) as pounds from
(select a.playerID, a.teamID, b.weight
from batting a
inner join Master b on a.playerID = b.playerID
group by a.playerID, a.teamID)
group by teamID
order by pounds desc

#Novel Thing 2: Find the average number of at_bats of teams in winning seasons versus losing seasons (Answer: 5173.66090712743 for winning season, 5093.932890855457 losing )

select avg(AB) as abswinning from
(select a.teamID, a.yearID, a.AB, case when a.W > a.L then 1 else 0 end as winningseason
from teams a)
where winningseason = 1

select avg(AB) as abswinning from
(select a.teamID, a.yearID, a.AB, case when a.W > a.L then 1 else 0 end as winningseason
from teams a)
where winningseason = 0

#Novel Thing 3: How many universities are players from?

select count(distinct schoolID) from schoolsplayers #(713)
select count(schoolID) from schoolsplayers #(6147)




#Novel Thing 4: Which player has the highest spread in votes needed and votes received for hall of fame inclusion and does this differ by different teams? (francju02	2013	ATL	421
#francju02	2013	MIL	421, nominated twice two different teams)




#Database 1: Number of Votes Over/Shy Hall of Fame Inclusion for each player, each year
select playerID, yearID, max(needed - votes) as gap
from halloffame
where needed is not null
group by playerID, yearID

#Database 2: 

select a.playerID, a.yearID, a.teamID, b.gap from batting a inner join
(select playerID, yearID, max(needed - votes) as gap
from halloffame
where needed is not null
group by playerID, yearID) b on a.playerID=b.playerID and a.yearID = b.yearID
group by a.playerID, a.yearID, a.teamID
order by gap desc



#Connecting to database using pandas

import sqlite3 as sq
import pandas as pd

#Query the baseball dataset

conn = sq.connect("C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite")

sql = """select avg(AB) as abswinning from
(select a.teamID, a.yearID, a.AB, case when a.W > a.L then 1 else 0 end as winningseason
from teams a)
where winningseason = 0"""

#passing the connection and SQL string
df = pd.read_sql(sql, conn)

#closing connection
conn.close()

#passing the connection and SQL string
df = pd.read_sql(sql, conn)

#closing connection
conn.close()

df['New'] = 'Test'

print df

conn = sq.connect("C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite")

df.to_sql('testingtable', conn, if_exists = 'replace')

conn.close


#Transform the data in some way

#Write a new table back to the database




















#eliminate the last year and then take max 

#Pass the SQL in the previous bullet into a pandas DataFrame and write it back to SQLite.

conn = sq.connect("C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite")

sql = """select avg(AB) as avgabs from
( select b.*, sq1.secyear from Batting b
left outer join
(select playerID, min(yearID) + 1 as secyear
from batting
group by playerID) sq1 on sq1.playerID = b.playerID and b.yearID = sq1.secyear)
where teamID = 'NYA' and secyear is not null and secyear > 1980"""


sql

test = pd.read_sql(sql, conn)

#closing connection
conn.close

test.head()

conn = sq.connect("C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite")

test.to_sql('testing', conn, if_exists = 'replace')

conn.close


#Four novel questions about the dataset



#1: # of Homeruns by league

SELECT b.lgID, sum(b.HR) as homeruns 
from Batting b
group by b.lgID
order by homeruns desc

#2: 