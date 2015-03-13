# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 18:52:56 2015

@author: jchen
"""

import numpy as np
import pandas
from statsmodels.formula.api import logit
from statsmodels.nonparametric import kde
import matplotlib.pyplot as plt
from patsy import dmatrix, dmatrices

df = pandas.read_csv('http://www.stat.columbia.edu/~gelman/arm/examples/arsenic/wells.dat', sep=' ', header=0, index_col=0)
print df.head()

# build model based only on distance to nearest safe well
# scale distance (increments of 100m v. 1), use I() indicator function to transform within the string
model1 = logit('switch ~ I(dist/100.)', data = df).fit() 
print model1.summary()

# create function that 'jitters' the binary values of 'switch' column for better visualization
def binary_jitter(x, jitter_amount = .05):
    '''
    Add jitter to a 0/1 vector of data for plotting.
    '''
    jitters = np.random.rand(*x.shape) * jitter_amount
    x_jittered = x + np.where(x == 1, -1, 1) * jitters
    return x_jittered

# First plot the Switch / No Switch dots vs distance to a safe well. Add jitter.
plt.plot(df['dist'], binary_jitter(df['switch'], .1), '.', alpha = .1)
# Now use the model to plot probability of switching vs distance (the green line).
sorted_dist = np.sort(df['dist']) # sort distribution
argsorted_dist = list(np.argsort(df['dist'])) # returns indices of sorted distribution
predicted = model1.predict()[argsorted_dist] # predict on index array
plt.plot(sorted_dist, predicted, lw = 2) # plot against sorted distance


# Plot the densities of distance for switchers and non-switchers
# We expect the distribution of switchers to have more mass over short distances 
# and the distribution of non-switchers to have more mass over long distances.
# kde = kernel density estimation
kde_sw = kde.KDEUnivariate(df['dist'][df['switch'] == 1])
kde_nosw = kde.KDEUnivariate(df['dist'][df['switch'] == 0])

kde_sw.fit()
kde_nosw.fit()

plt.plot(kde_sw.support, kde_sw.density, label = 'Switch')
plt.plot(kde_nosw.support, kde_nosw.density, color = 'red', label = 'No Switch')
plt.xlabel('Distance (meters)')
plt.legend(loc = 'best')


# Add the arsenic level as a regressor.
# We'd expect respondents with higher arsenic levels to be more motivated to switch.
model2 = logit('switch ~ I(dist / 100.) + arsenic', data = df).fit()
print model2.summary()

# To see the effect of these on the probability of switching, 
# let's calculate the marginal effects at the mean of the data.
margeff =  model2.get_margeff(at = 'mean')
print margeff.summary()

# Class separability
# To get a sense of how well this model might classify switchers and non-switchers,
# we can plot each class of respondent in (distance-arsenic)-space.
logit_pars = model2.params # coefficients 
intercept = -logit_pars[0] / logit_pars[2] 
slope = -logit_pars[1] / logit_pars[2] 

dist_sw = df['dist'][df['switch'] == 1]
dist_nosw = df['dist'][df['switch'] == 0]
arsenic_sw = df['arsenic'][df['switch'] == 1]
arsenic_nosw = df['arsenic'][df['switch'] == 0]
plt.figure(figsize = (12, 8))
plt.plot(dist_sw, arsenic_sw, '.', mec = 'purple', mfc = 'None', 
         label = 'Switch')
plt.plot(dist_nosw, arsenic_nosw, '.', mec = 'orange', mfc = 'None', 
         label = 'No switch')
plt.plot(np.arange(0, 350, 1), intercept + slope * np.arange(0, 350, 1) / 100.,
         '-k', label = 'Separating line')
plt.ylim(0, 10)
plt.xlabel('Distance to safe well (meters)')
plt.ylabel('Arsenic level')
plt.legend(loc = 'best')


# Model 3: Adding an interaction
# It's sensible that distance and arsenic would interact in the model. 
# In other words, the effect of an 100 meters on your decision to switch would be affected by how much arsenic is in your well.
# We don't have to pre-compute an explicit interaction variable. We can just specify an interaction in the formula description using the : operator.
model3 = logit('switch ~ I(dist / 100.) + arsenic + I(dist / 100.):arsenic', 
                   data = df).fit()
print model3.summary()



