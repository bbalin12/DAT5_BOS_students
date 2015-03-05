# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 09:07:08 2015

@author: jeppley
"""

import numpy as np
import pandas
from statsmodels.formula.api import logit
from statsmodels.nonparametric import kde
import matplotlib.pyplot as plt
from patsy import dmatrix, dmatrices
import sqlite3 as sql


######################################################################
###Using the baseball dataset, build a logistic regression model 
###that predicts who is likely to be inducted into Hall of Fame.
######################################################################


########################################
###  Extracting Data prior to 2000   ###
########################################

con = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')

data = '''select h.*, 
  p.*
from 
  (select playerid, inducted
  from hall_of_fame_inductees_3 
   where category = 'Player'
   group by playerid) h
inner join 
  (select playerid,
    count(distinct yearid) as p_years,
    sum(w) as p_wins,
    sum(l) as p_loss,
    sum(sho) as p_shout,
    sum(sv) as p_saves,
    sum(er) as p_eruns,
    sum(so) as p_stout
  from pitching
  group by playerid) p
  on h.playerid = p.playerid;'''
  
new = pandas.read_sql(data, con)
con.close()




##############################################################
###  Start with considering as many explanatory variables  ###
##############################################################

model1 = logit('inducted ~ p_wins + p_loss + p_shout + p_saves + p_stout', data = new).fit() # model1 is our fitted model.
print model1.summary()


#==============================================================================
#                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
#------------------------------------------------------------------------------
#Intercept     -2.3188      0.310     -7.477      0.000        -2.927    -1.711
#p_wins         0.0273      0.006      4.377      0.000         0.015     0.039
#p_loss        -0.0346      0.007     -4.957      0.000        -0.048    -0.021
#p_shout        0.0193      0.022      0.886      0.376        -0.023     0.062
#p_saves    -5.344e-06      0.004     -0.001      0.999        -0.008     0.008
#p_stout        0.0005      0.000      1.430      0.153        -0.000     0.001
#==============================================================================


#############################################################################
###  Reduce your explanatory variables to the ones that are significant.  ###
#############################################################################

model2 = logit('inducted ~ p_wins + p_loss + p_stout', data = new).fit() 
print model2.summary()


#==============================================================================
#                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
#------------------------------------------------------------------------------
#Intercept     -2.3559      0.296     -7.948      0.000        -2.937    -1.775
#p_wins         0.0309      0.005      6.198      0.000         0.021     0.041
#p_loss        -0.0363      0.007     -5.320      0.000        -0.050    -0.023
#p_stout        0.0006      0.000      1.916      0.055     -1.34e-05     0.001
#==============================================================================


#####################################################################################
###  Cross validate your model and print out the coeffecients of the best model.  ###
#####################################################################################

########################################
###  Extracting Data after 2000   ###
########################################

con = sql.connect('C:\Users\jeppley\datascience\SQLite\lahman2013.sqlite')

data = '''select h.*, 
  p.*
from 
  (select playerid, inducted
  from hall_of_fame_inductees_post20003 
   where category = 'Player'
   group by playerid) h
inner join 
  (select playerid,
    count(distinct yearid) as p_years,
    sum(w) as p_wins,
    sum(l) as p_loss,
    sum(sho) as p_shout,
    sum(sv) as p_saves,
    sum(er) as p_eruns,
    sum(so) as p_stout
  from pitching
  group by playerid) p
  on h.playerid = p.playerid;'''
  
new_2000 = pandas.read_sql(data, con)
con.close()




#####################################################
###  Cross-validating model on post 2000 data  ###
#####################################################

model3 = logit('inducted ~ p_wins + p_loss + p_stout', data = new_2000).fit() 
print model3.summary()

#==============================================================================
#                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
#------------------------------------------------------------------------------
#Intercept     -3.2566      0.888     -3.667      0.000        -4.997    -1.516
#p_wins        -0.0012      0.017     -0.068      0.946        -0.035     0.033
#p_loss         0.0014      0.017      0.082      0.935        -0.031     0.034
#p_stout        0.0005      0.001      0.408      0.684        -0.002     0.003
#==============================================================================


#validation presents concerns about the model and the use of wins/losses, likely
#as it is a time related variable and not enough win loss data ended up being
#available after 2000
#evidence to chuck out wins and losses
#or create a win/loss ratio variable







new_2000['win_loss'] = (new_2000['p_wins']/new_2000['p_loss'])

new_2000.tail()
#appears there are a number of cases where wins and losses are 0, want to remove those

new_2000 = new_2000['p_wins'] is not 0


new_2000 = new_2000.query('p_wins != 0')

new_2000.describe()


#==============================================================================
#                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
#Intercept     -5.9305      1.679     -3.532      0.000        -9.222    -2.639
#p_wins        -0.0128      0.018     -0.691      0.489        -0.049     0.023
#p_loss         0.0236      0.019      1.230      0.219        -0.014     0.061
#p_stout        0.0011      0.001      0.925      0.355        -0.001     0.003
#==============================================================================
#Removing the cases of 0's improves things but not completely

new_2000['win_loss'] = (new_2000['p_wins']/new_2000['p_loss'])
new_2000['loss_win'] = (new_2000['p_loss']/new_2000['p_wins'])

model3b = logit('inducted ~ loss_win + p_stout', data = new_2000).fit() 
print model3b.summary()
#although the ratio becomes more significant than just raws wins/losses
#still appears that within the subset that doesn't make much difference

#####################################################
###  Removing wins and losses and using only stats  ###
#####################################################

model4 = logit('inducted ~ p_shout + p_saves + p_stout', data = new).fit() 
print model4.summary()



#==============================================================================
#                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
#------------------------------------------------------------------------------
#Intercept     -2.5369      0.285     -8.887      0.000        -3.096    -1.977
#p_shout        0.0571      0.014      4.060      0.000         0.030     0.085
#p_saves        0.0017      0.004      0.443      0.658        -0.006     0.009
#p_stout    -5.347e-05      0.000     -0.198      0.843        -0.001     0.000
#==============================================================================



#####################################################################################
###  Considering any two features, generate a scatter plot with a class separable line showing the classification.  ###
#####################################################################################


def binary_jitter(x, jitter_amount = .05):
    '''
    Add jitter to a 0/1 vector of data for plotting.
    '''
    jitters = np.random.rand(*x.shape) * jitter_amount
    x_jittered = x + np.where(x == 1, -1, 1) * jitters
    return x_jittered
    
    
# First plot the Inducted / Not Indcuted dots vs p_shout. Add jitter.
plt.plot(new['p_shout'], binary_jitter(new['inducted'], .1), '.', alpha = .1)
# Now use the model to plot probability of induction vs shut outs (the green line).
sorted_dist = np.sort(new['p_shout'])
argsorted_dist = list(np.argsort(new['p_shout']))
predicted = model4.predict()[argsorted_dist]
plt.plot(sorted_dist, predicted, lw = 2)



















