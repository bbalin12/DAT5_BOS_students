# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:24:57 2015

@author: Margaret
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor
import numpy as np
import os
import glob
import itertools

from pylab import rcParams
from scipy.stats import itemfreq

#os.chdir(os.path.dirname('__file__'))
os.path.dirname(os.path.realpath('__file__'))

path = '/Users/Margaret/Desktop/data_science/general_assembly/project/'
#path =''

## USA DATA
files = glob.glob(path+'data/*/yfcc100m_dataset-?.*')

### FUNCTIONS
## Process each Dataframe
def process_df(filename):
    """
    Processes the file based on the fields.
    Flickr fields are:
    * Photo/video ID, User NSID, User nickname, Date taken, Date uploaded, Capture device
    * Title, Description, User tags (comma-separated), Machine tags (comma-separated)
    * Longitude, Latitude, Accuracy, Photo/video page URL, Photo/video download URL
    * License name, License URL, Photo/video server identifier, Photo/video farm identifier
    * Photo/video secret, Photo/video secret original, Photo/video extension original
    * Photos/video marker (0 = photo, 1 = video)
    
    Keeps date taken, camera, longitude, and latitutde.
    """
    df = pd.read_table(filename, sep='\t', 
                       names=['photovidID','userID','username', 'date_taken', 'date_uploaded', 
                              'camera','title','desc','tags','mach_tags','long', 'lat', 
                              'accuracy','page_url','dl_url', 'license','lic_url',
                              'serverID','farmID','secret','secret_orig','ext_orig','photo_vid'],
                                  index_col=False, skipinitialspace = True, nrows = 50000)
    df.dropna(axis=0,subset=['camera','long','lat'], inplace=True)
    df = df[['date_taken','camera','long','lat']]
    return df

## Append Dataframes to a list of Dataframes
def create_fulldf(files):
    """
    Appends each dataframe created to a master dataframe
    """
    flickr_dfs = []
    for f in files:
        flickr_dfs.append(process_df(f))
    return flickr_dfs

# Removing entries with a specific word from result
def wordinentry(df, column, word):
    """
    Finds if a word is in an entry in a specific column of the dataframe.
    """
    bool_series = []
    in_entry = False    
    for entry in df[column]:
        if word in entry.lower():
            in_entry = True
        else:
            in_entry = False
        #print entry, in_entry
        bool_series.append(in_entry)
    return bool_series

# Cleanup/bin data that does not occur a lot
def cleanup_data(df, col, cutoffPercent):
    """
    Bins data that appears less often than the cutoffPercent
    """
    sizes = df[col].value_counts(normalize = True)
    # get the names of the levels that make up less than the cutoff % of the dataset
    values_to_delete = sizes[sizes<cutoffPercent].index
    df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df


def splitstring(entry,splits):
    """
    For every character in splits, splits the entry string.
    """
    word = ''
    result = []
    for char in entry:
        if splits.find(char) != -1 and word != '':
            result.append(word)
            word = ''
        elif splits.find(char) == -1:
            word += char
    if word != '':
        result.append(word)
    return result


def itemfreq(df, col):
    """
    Finds an item in a column and finds how frequently it occurs.
    """
    label=[]
    freq=[]
    for item in itemfreq(df[col]):
        label.append(item[0])
        freq.append(item[1])
    return label, freq

######################
### CREATING DATAFRAME
######################

# Set option to display all columns
pd.set_option('display.max_columns', None)

# Change size of graph/plot
rcParams['figure.figsize'] = 16, 9 
rcParams['agg.path.chunksize'] = 20000

# Create full Dataframe by concatenating list of Dataframes
flickr_df = pd.concat(create_fulldf(files))

# Reset index
flickr_df.reset_index(drop=True, inplace=True)

#######################
### MODIFYING DATAFRAME
#######################


# Taking out scanners that are listed as cameras
wordnotinentry = np.invert(wordinentry(flickr_df,'camera','scan'))
flickr_df = flickr_df[wordnotinentry]

# Create a new column for camera brand
flickr_df['camera_brand'] = flickr_df.camera.map(lambda x: x.split('+')[0])
flickr_df['camera_brand'] = flickr_df.camera_brand.map(lambda x: x.split('%')[0])
flickr_df['camera_brand'] = flickr_df.camera_brand.map(lambda x: x.split('_')[0])

# Recategorize the brands
flickr_df = cleanup_data(flickr_df,'camera_brand',0.01)


############################
### WRITING OUT AND PLOTTING
############################
    
# Write to a .csv file
#flickr_df = flickr_df.sort(columns=['date_taken'])
#flickr_df.to_csv(path_or_buf='/Users/Margaret/Desktop/data_science/general_assembly/project/data/latlongdata.csv')

flickr_df = flickr_df[(flickr_df.date_taken < '2015-04-01 00:00:00.0')
            & (flickr_df.date_taken > '1850-01-01 00:00:00.0')]

flickr_df = flickr_df.sort(columns=['date_taken'])
flickr_df.reset_index(drop=True, inplace=True)


### SETTING COLORMAP
# http://flatuicolors.com/
color_scheme = [
                (213, 94, 0.0), #orange
                (0.0, 158, 115), #turquoise
                (230, 126, 34), #carrot
                (39, 174, 96), #darker emerald
                (192, 57, 43), # pomegranate
                (52, 152, 219), #blue
                (189, 195, 199), #silver
                (155, 89, 182), # amethyst
                (127, 140, 141), # grey
                (25, 25, 112), # dark blue grey
                (179, 222, 105), # light green
                (128, 0.0, 128), #purple
                (241, 196, 15), #sunflower
                (166, 60, 40), #berry
                (255, 237, 111), #bright yellow
                (178, 34, 34), #firebrick
                ] 
                
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(color_scheme)):  
    r, g, b = color_scheme[i]  
    color_scheme[i] = (r / 255., g / 255., b / 255.) 

# Create color iterator    
color_cycle = itertools.cycle(color_scheme)

# Create color map
cmap = pltcolor.ListedColormap(color_scheme)

######################
### BREAKING DOWN DATA
######################

# Plot overview of data
plt.scatter(flickr_df.long, flickr_df.lat, s=3)
plt.title("Flickr in the United States and Northern Central America")
plt.xlim([-200, -40])
plt.ylim([-10,80])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig(path+"images/us.png")

### CAMERA BRANDS

# developing a dict of photograph years and brands and frequency of brands
years = range(2000,2015)
brandfreq_dict = {}
for year in years:
    brands_freqs = itemfreq(flickr_df[(flickr_df.date_taken < (str(year+1)+'-01-01 00:00:00.0'))
            & (flickr_df.date_taken > (str(year)+'-01-01 00:00:00.0'))].camera_brand)
    brand = brands_freqs[:,0]
    freq = brands_freqs[:,1]
    brandfreq_dict[year] = brands_freqs
    
    fig, ax = plt.subplots()
    ax.bar(xrange(len(brand)),freq, align = 'center', label=brand, color = color_scheme[-3])
    ax.set_xlim(-1, len(brand)+0.5)
    ax.set_ylim([0,70000])
    ax.set_xticks(range(len(brand)))
    ax.set_xticklabels(brand, ha='right',rotation_mode='anchor',rotation = 45)
    ax.set_title(str(sum(brandfreq_dict[year][:,1])) + " Photos by Camera Brand in " + str(year))
    ax.set_ylabel("Number of Photos")
    ax.set_xlabel("Camera Brands")
    fig.savefig(path+"images/camerabrands" + str(year) + ".jpg")

brands_dict = {}
for each_brand in brand:
    years_brand=[]
    for year in years:
        index_brand = np.where(brandfreq_dict[year][:,0] == each_brand)
        freq_brand = brandfreq_dict[year][index_brand[0],1]
        if freq_brand.size == 0:
            freq_brand = 0
        else:
            freq_brand = freq_brand[0]
        years_brand.append([year, freq_brand])
        brands_dict[each_brand] = years_brand


######################
### K-MEANS CLUSTERING
######################
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Evaluate K-Means Models based on Silhouette Coefficient to Optimize
best_clusters = range(2,31)
n_samples = 10000

kmeans_sscore = {}

for bc in best_clusters:
    kmeans_bc = KMeans(n_clusters = bc)

    # Fit KMeans Model
    kmeans_bc.fit(flickr_df[['long','lat']])
    labels_bc = kmeans_bc.labels_
    #cluster_centers_bc = kmeans_bc.cluster_centers_
    
    flickr_array = flickr_df.as_matrix(columns = ['long','lat'])
    kmeans_sscore[bc] = silhouette_score(flickr_array, labels_bc, metric = 'euclidean', sample_size = n_samples)
    print "Silhouette Score for K-Means Using " + str(bc) + " Clusters: %0.3f" % kmeans_sscore[bc]

color = next(color_cycle)
fig, ax = plt.subplots()
ax.scatter(kmeans_sscore.keys(), kmeans_sscore.values(), c = color)
ax.plot(kmeans_sscore.keys(), kmeans_sscore.values(), c = color, lw=1.5)
ax.plot(15,kmeans_sscore[15], 'ro', markersize=14, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor=next(color_cycle))
ax.set_title("Optimal Clusters for K-Means Model")
ax.text(15,kmeans_sscore[15]+0.025, "Chosen # of Clusters")
ax.set_xlim(best_clusters[0], best_clusters[-1])
ax.set_ylim([0.5,0.9])
ax.set_xlabel("Clusters")
ax.set_ylabel("Silhouette Score")
fig.savefig(path+"images/kmeans_silhouette.jpg")


### CREATE OPTIMAL KMEANS MODEL    
clusters = 15
cluster_index = range(0,clusters)

kmeans_est = KMeans(n_clusters = clusters)

# Fit KMeans Model
kmeans_est.fit(flickr_df[['long','lat']])
labels = kmeans_est.labels_
cluster_centers = kmeans_est.cluster_centers_ 


# Plots all the points, colored by cluster
fig, ax = plt.subplots()
colors = cmap(labels)
ax.scatter(flickr_df.long, flickr_df.lat, s=30, c=colors)
ax.set_title("Flickr Photos Colored in " + str(clusters) + " Clusters")
ax.set_xlim([-200, -40])
ax.set_ylim([-10,80])
ax.set_ylabel("Latitude")
ax.set_xlabel("Longitude")
fig.savefig(path+"images/kmeans_" + str(clusters) + "clusters.jpg")

# Plots the cluster centroids
fig, ax = plt.subplots()
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], marker = 'o', s=200, c='w')
for i, c in enumerate(cluster_centers):
    ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=80)
    ax.set_title("Cluster Centroids for " + str(clusters) + " Clusters")
    ax.set_xlim([-200, -40])
    ax.set_ylim([-10,80])   
    ax.set_ylabel("Latitude")
    ax.set_xlabel("Longitude")
fig.savefig(path+"images/kmeans_" + str(clusters) + "clustercenters.jpg")


### Determining all points in each cluster
cluster_points = {}
longlat_df = flickr_df[["long","lat"]]
for cluster in cluster_index:
    cluster_points[cluster] = longlat_df[labels==cluster] #clusternum

### Creating dataframe of number of points in each cluster by year
### Creating dataframe of the points in each cluster by year
years_all = {} # keys are each year, values are dataframe of that year
cluster_years = {} # keys are each year, values are cluster, values in each cluster are # of points
clustpts_index = {}
years = range(2000,2015)
counter = 0

for year in years:
    cluster_years[year] = {}
    clustpts_index[year] = {}
    years_all[year] = flickr_df[(flickr_df.date_taken <= str(year+1)+'-01-01 00:00:00.0') & 
                        (flickr_df.date_taken >= str(year)+'-01-01 00:00:00.0')][['long','lat']]
    for cluster in cluster_index:        
        counter = 0        
        ptsincluster = []
        ptsindex = []
        index = years_all[year].index
        for each in index:
            if each in cluster_points[cluster].index:
                counter += 1
                ptsincluster.append(years_all[year].loc[each])
                ptsindex.append(each)
        clustpts_index[year][cluster] = ptsindex
        cluster_years[year][cluster] = counter



### Plot all maps of the clusters
for year in years:
    fig, ax = plt.subplots()
    regions = []
    for clusts in clustpts_index[year]:
        colors = cmap(labels[clustpts_index[year][clusts]])
        x = flickr_df.iloc[clustpts_index[year][clusts]].long
        y = flickr_df.iloc[clustpts_index[year][clusts]].lat
        regions.append(ax.scatter(x, y, marker = 'o', s=30, c=colors))
    
        ax.set_title(str(sum(cluster_years[year].values())) + " Photos in " + str(clusters) + " Clusters in " + str(year))
        ax.set_xlim([-200, -40])
        ax.set_ylim([-10,80])
        ax.set_ylabel("Latitude")
        ax.set_xlabel("Longitude")
    fig.savefig(path+"images/clusters_" + str(year) + "color.jpg")

############################################
### PREDICTING VALUES WITH LINEAR REGRESSION
############################################
import statsmodels.formula.api as smf


# Create dataframe of number of points in each cluster per year
clusteryears_expdf = pd.DataFrame.from_dict(cluster_years, orient='columns')

rsq = [0]*len(clusteryears_expdf.index)
RMSE = [0]*len(clusteryears_expdf.index)
RMSE_perc = [0]*len(clusteryears_expdf.index)
forecast_years = 6
test_clustery = {}

for c in clusteryears_expdf.index:
    cluster_i = clusteryears_expdf.ix[c,:]
    cluster_i = cluster_i[0:len(cluster_i)-1] # removes 2014 since it is more incomplete
    
    train_cluster = pd.DataFrame(columns=('x1', 'x2', 'x3', 'x4', 'x5', 'y'))
    
    for i in range(0,(len(cluster_i) - 5)):
       train_cluster.loc[i] = [cluster_i.iloc[i], cluster_i.iloc[i+1], 
                            cluster_i.iloc[i+2], cluster_i.iloc[i+3], 
                            cluster_i.iloc[i+4], cluster_i.iloc[i+5]]
                            
    ### LINEAR REGRESSION MODEL
    est = smf.ols(formula = "y ~ x1 + x2 + x3 + x4 + x5", data = train_cluster).fit()
    est.summary()

    rsq[c] = est.rsquared
    
    # prediction and linear extrapolation of training data set to get further predictions.
    test_cluster = train_cluster.copy()
    
    for i in range(0,(len(cluster_i) - 5)):
       test_cluster.loc[i] = [cluster_i.iloc[i], cluster_i.iloc[i+1], 
                            cluster_i.iloc[i+2], cluster_i.iloc[i+3], 
                            cluster_i.iloc[i+4],
                            est.predict(test_cluster.ix[i,0:5])[0]]
    
    # further running time series to predict into the future
    j = len(test_cluster) - 1
    for i in range(j, j+forecast_years):
       test_list = test_cluster.ix[i,1:6].tolist()
       y_est = est.predict(test_cluster.ix[i,0:5])
       test_list.append(y_est[0])
       test_series = pd.Series(test_list, index = train_cluster.columns)
       test_cluster = test_cluster.append(test_series, ignore_index = True)
    
    test_clustery[c] = test_cluster['y']
    residuals = test_cluster['y'][0:len(train_cluster)] - train_cluster['y']
    
    RMSE[c] = (((residuals)**2).mean())**(0.5)
    RMSE_perc[c] = RMSE[c]/(test_cluster['y'][0:len(train_cluster)].mean())*100
    
    ### PLOTTING
    lastyear = years[-1] + forecast_years
    x = range(years[0], lastyear)  
    y1 = np.concatenate([train_cluster.ix[0,range(len(train_cluster.columns)-1)], train_cluster['y']])
    y2 = np.concatenate([test_cluster.ix[0,range(len(train_cluster.columns)-1)], test_cluster['y']]) 
    
    fig, ax = plt.subplots()
    color1 = next(color_cycle)
    color2 = next(color_cycle)
    traindot = ax.scatter(x[0:len(y1)], y1, c=color1, s=60, alpha = 0.7)
    trainlin = ax.plot(x[0:len(y1)], y1, c=color1, lw=2)
    testdot = ax.scatter(x, y2, c=color2, s=60, marker="o", alpha = 0.7)
    testlin = ax.plot(x, y2, c=color2, lw=2)
    testlast = ax.plot(x[-1],y2[-1], 'ro', markersize=14, markeredgewidth=1.5,
         markerfacecolor='None', markeredgecolor=color2)
    plt.plot([2013, 2013], [-12000, 60000], 'k--', lw=1, alpha = 0.5)
    ax.set_title("Predictions of Photos in Cluster #" + str(c))
    ax.legend(['Actual', 'Prediction'])
    ax.text(x[-1],y2[-1]+1500, int(y2[-1]))
    ax.text(0.025, 0.95, "R-Squared: %0.2f" %(rsq[c]), ha = "left", va = "center", transform = ax.transAxes, size = 12)
    ax.text(0.025, 0.91, "RMSE: %0.1f" %(RMSE_perc[c]) + "%", ha = "left", va = "center", transform = ax.transAxes, size = 12)
    ax.set_xlim([2000, 2020.5])
    ax.set_ylim([-7500,60000])
    ax.set_xlabel("Year")
    ax.set_ylabel("Photos")
    fig.savefig(path+"images/prediction_cluster" + str(c) + ".jpg")
