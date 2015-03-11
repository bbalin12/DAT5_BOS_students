-- Create full, working queries to answer at least four novel questions you have about the dataset

-- Which players were inducted into the Hall Of Fame with the most favor?
select nameGiven as player_name,
	f.yearID as year_inducted,
	round(votes*100.0/ballots,2) as percent_vote
from HallOfFame f
inner join Master m on f.playerID=m.playerID
where f.votes is not null
	and f.ballots is not null
	and f.inducted = 'Y'
group by player_name
order by percent_vote desc
limit 10

-- Count number of players who have the same batting hand as throwing hand
select case when bats='B' then 'both'
			when bats=throws then 'same'
	   		else 'different'
   	   end as same_hand,
   	   count(distinct playerID) as player_count
from Master
group by same_hand
order by player_count desc

-- Which teams have had the longest and shortest existence?
with team_years as
	(select name as team_name,
		count(distinct yearID) as years
	from Teams
	group by 1)
select *
from team_years
where years in (select max(years) from team_years)
        or years in (select min(years) from team_years)
order by years, team_name

-- Which players have the best OPS (on-base percentage + slugging average) in a season with at least 300 at bats?
-- OPS = OBP + SLG
-- where OBP = (H+BB+HBP)/(AB+BB+SF+HBP)
-- and SLG = TB/AB
-- Total Bases TB = H+2B+(3B*2)+(HR*3)
with performance as
	(select playerID,
		yearID,
		teamID,
		(H+BB+HBP)*1.0/(AB+BB+SF+HBP) as OBP,
		(H+"2B"+("3B"*2)+(HR*3))*1.0/AB as SLG
	from Batting
	where AB>300
	group by 1,2,3)
select m.nameGiven as player_name,
	p.yearID,
	t.name as team_name,
	round(OBP+SLG,2) as OPS
from performance p
left join Master m on p.playerID=m.playerID
left join Teams t on p.teamID=t.teamID and p.yearID=t.yearID
group by 1,2,3
order by OPS desc
limit 10






