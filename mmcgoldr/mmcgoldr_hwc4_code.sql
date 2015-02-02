--Megan McGoldrick
--GA Data Science Class 4 HW

--Average at bats among Yankees players who began their second season on/after 1980 = 197.7
--STEPS:
--A. Find first year played for Yankees
--B. Find second year played for Yankees 
--C. Exludes players with: 1) only one season 2) second Yankees season before 1980 (assumes second Yankee season may not be same as second career season)
--D. Pull in all At Bats with Yankees for these players and calcuate average (excludes At Bats for other teams)

select avg(d.ab) as NYA_AB_season2_after1979
from
  (select b.playerid, a.first_year_NYA, min(b.yearid) as second_year_NYA
   from
     (select playerid, min(yearid) as first_year_NYA from batting where teamid = 'NYA' group by playerid) a
   inner join
     (select playerid, yearid from batting where teamid = 'NYA') b
   on a.playerid = b.playerid and a.first_year_NYA != b.yearid
   group by b.playerid
   having second_year_NYA >= 1980) c
inner join
  (select playerid, ab from batting where teamid = 'NYA') d
on c.playerid = d.playerid

--Which teams have had the highest number of unique players play in an allstar game in last 10 years?
--Answer: DET, SLN, TEX (18 each)
select teamid, count(distinct playerid) as distinct_players
from allstarfull 
where yearid > 2003 and gp = 1
group by teamid
order by distinct_players desc

--Name the player who has won the most number of World Series
--Answer: Yogi Berra (10 wins)
select e.playerid, g.nameFirst, g.nameLast, count(*) as ws_wins
from 
  (select teamid, yearid from teams where wswin = 'Y' order by yearid desc) f

left join
  (select d.teamid, d.yearid, d.playerid
   from
     (select a.* from (select playerid, yearid, teamid from batting group by playerid, yearid, teamid) a
      union all
      select b.* from (select playerid, yearid, teamid from pitching group by playerid, yearid, teamid) b 
      union all
      select c.* from (select playerid, yearid, teamid from fielding group by playerid, yearid, teamid) c) d
   group by d.teamid, d.yearid, d.playerid) e
on f.teamid = e.teamid and f.yearid = e.yearid

left join 
  master g 
on e.playerid = g.playerid

group by e.playerid
order by ws_wins desc

--Does pitching performance differ by throwing hand (left or right)?
--Answer: On average, right-handed pitchers have slighter higher wins and just slightly lower ERA than left-handed pitchers
select m.throws, avg(g) as avg_g, avg(gs) as avg_gs, avg(w) as avg_wins, avg(era) as avg_era
from pitching p
left join master m
on p.playerid = m.playerid
where p.era IS NOT NULL and m.throws IS NOT NULL
group by m.throws

--How does pitching performance differ among frequent vs infrequent starters?
--Average ERA is about 20% better among frequent starts (3.98) vs infrequent starter (5.02)
select c.frequent_gs, avg(c.era) as avg_era
from
  (select a.playerid, case when a.starts_per_year > 20 then 1 else 0 end as frequent_gs, b.w, b.era
   from
     (select playerid, sum(gs)/count(distinct yearid) as starts_per_year 
      from pitching 
      group by playerid 
      having starts_per_year > 1) a
   left join 
     pitching b 
   on a.playerid = b.playerid) c
group by c.frequent_gs