-- Find the player with the most at-bats in a single season.
select player_name
from
(select nameGiven as player_name,
	yearID,
	sum(AB) as total_at_bats
from batting b
left join master m on b.playerID=m.playerID
group by 1,2
order by 3 desc
limit 1)

-- Find the name of the the player with the most at-bats in baseball history.
select player_name 
from
(select nameGiven as player_name,
	sum(AB) as historical_at_bats
from batting b
left join master m on b.playerID=m.playerID
group by 1
order by 2 desc
limit 1) as a

-- Find the average number of at_bats of players in their rookie season.
with rookies as 
	(select playerID, min(yearID) as rookie_year from batting b group by 1)
select avg(AB) as avg_at_bats
from batting b
inner join rookies r on b.playerID=r.playerID
	and b.yearID=r.rookie_year

-- Find the average number of at_bats of players in their final season for all players born after 1980.
with finals as 
	(select b.playerID, max(b.yearID) as final_year from batting b left join master m on b.playerID=m.playerID where m.birthYear > 1980 group by 1)
select avg(AB) as avg_at_bats
from batting b
inner join finals f on b.playerID=f.playerID
	and b.yearID=f.final_year


-- Find the average number of at_bats of Yankees players who began their second season at or after 1980.
with yankee_rookies as 
	(select playerID, min(b.yearID) as rookie_year 
	from batting b 
	inner join teams t on b.teamID=t.teamID and b.yearID=t.yearID
	where t.name='New York Yankees' 
	group by 1),
	yankee_years as
	(select b.playerID, b.yearID
	from batting b 
	inner join teams t on b.teamID=t.teamID and b.yearID=t.yearID
	where t.name='New York Yankees'
	group by 1,2),
	yankee_seconds as
	(select playerID, min(yearID) as second_year
	from (select * from yankee_years except select * from yankee_rookies) as a
	group by 1)
select avg(b.AB) as avg_at_bats
from batting b
inner join teams t on b.teamID=t.teamID
where playerID in (select playerID from yankee_seconds where second_year>=1980)
	and t.name='New York Yankees'
