[cb2004]: /images/camerabrands2004.jpg
[cb2005]: /images/camerabrands2005.jpg
[cb2006]: /images/camerabrands2006.jpg
[cb2007]: /images/camerabrands2007.jpg
[cb2008]: /images/camerabrands2008.jpg
[cb2009]: /images/camerabrands2009.jpg
[cb2010]: /images/camerabrands2010.jpg
[cb2011]: /images/camerabrands2011.jpg
[cb2012]: /images/camerabrands2012.jpg
[cb2013]: /images/camerabrands2013.jpg
[cb2014]: /images/camerabrands2014.jpg

[us]: /images/us.jpg
[kmclusters]: /images/kmeans_15clusters.jpg
[kmcenters]: /images/kmeans_15clustercenters.jpg
[kmsil]: /images/kmeans_silhouette.jpg

[cl2000]: /images/clusters_2000color.jpg
[cl2001]: /images/clusters_2001color.jpg
[cl2002]: /images/clusters_2002color.jpg
[cl2003]: /images/clusters_2003color.jpg
[cl2004]: /images/clusters_2004color.jpg
[cl2005]: /images/clusters_2005color.jpg
[cl2006]: /images/clusters_2006color.jpg
[cl2007]: /images/clusters_2007color.jpg
[cl2008]: /images/clusters_2008color.jpg
[cl2009]: /images/clusters_2009color.jpg
[cl2010]: /images/clusters_2010color.jpg
[cl2011]: /images/clusters_2011color.jpg
[cl2012]: /images/clusters_2012color.jpg
[cl2013]: /images/clusters_2013color.jpg
[cl2014]: /images/clusters_2014color.jpg

[Pacific Northwest]: /images/prediction_cluster0.jpg
[Mid-Atlantic]: /images/prediction_cluster1.jpg
[Hawaii]: /images/prediction_cluster2.jpg
[South]: /images/prediction_cluster3.jpg
[Alaska]: /images/prediction_cluster4.jpg
[Southwest]: /images/prediction_cluster5.jpg
[Central America]: /images/prediction_cluster6.jpg
[Northern Mountains]: /images/prediction_cluster7.jpg
[Great Lakes]: /images/prediction_cluster8.jpg
[Southeast]: /images/prediction_cluster9.jpg
[California]: /images/prediction_cluster10.jpg
[Northeast]: /images/prediction_cluster11.jpg
[Caribbean]: /images/prediction_cluster12.jpg
[Rocky Mountains]: /images/prediction_cluster13.jpg
[Western Canada]: /images/prediction_cluster14.jpg


# Predicting Travel Patterns Using Flickr
Using the 100 Million Photos and Videos database from Flickr to predict travel patterns within the United States and Central America.

## The Hypothesis:

People typically travel to take photographs, or go to a specific place to take photographs. Even if it is their backyard, it is a place that has meaning and visual attraction. I am interested in looking at photography as a predictor of ideal locations to travel to. Where do people like to take photographs? Where _will_ people like to take photographs?

#### *Where will people travel?*



## The Preprocessing/Cleaning/Manipulation

The Flickr database consists of the following: 

- Photo/video ID
- User NSID, User nickname
- Date taken
- Date uploaded
- Capture device
- Title, Description
- User tags (comma-separated), Machine tags (comma-separated)
- Longitude, Latitude
- Accuracy
- Photo/video page URL, Photo/video download URL
- License name, License URL
- Photo/video server identifier, Photo/video farm identifier
- Photo/video secret, Photo/video secret original
- Photo/video extension original
- Photos/video marker (0 = photo, 1 = video)

Cleaning consisted of the following steps:
- Taking out any cameras with "scan" in the name
- Binning the rest of the camera brands, putting any that occur less than 1% of the time into a category "Other"


## Visual Explorations

Through explorations of the camera brands apparent in the dataset, it is clear that there is a growth of Canon cameras over time, although the introduction of the Apple iPhone in 2007 quickly brings Apple into the ring for contention. 

![cb2006]                         
![cb2007]


## Clustering Optimization and Analysis

This analysis focused on the United States and Central America, and K-Means Clustering was used to break up the area into regions. To develop the optimal number of clusters, a silhouette score was assigned to a range of clusters. Using the scores as a guideline, the final number of clusters selected was 15. 

![kmsil]
![kmclusters]


## Linear Regression

The points were grouped into each sluter, and used that to create the set of time series below, sorted by region. On average, the R-squared values were 86.2%, with a root mean square error of 11.9%, using a time-slice of five years to predict each sixth year. 


Far West              |West                   | Central              | East                
:--------------------:|:---------------------:|:--------------------:|:--------------------:
[Alaska]              |[Pacific Northwest]    |[Northern Mountains]  |[Northeast]
[Western Canada]      |[California]           |[Rocky Mountains]     |[Mid-Atlantic]
[Hawaii]              |[Southwest]            |[Great Lakes]         |[Southeast]
                      |[Central America]	  |[South]               |[Caribbean]




# What Will Happen in 2019?

Based on the analysis, the __Pacific Northwest__ will be the most popular place, holding its status from 2000 onward. The least popular locations will be Hawaii and the South. There will be a growing trend in visits to Central America, and to California.

_Pacific Northwest_
![Pacific Northwest]

_Central America_
![Central America]

_California_
![California]

_Hawaii_
![Hawaii]


## Next Steps

This has been based on a simple K-Means clustering, with the number of clusters fine tuned. It also has been sliced into a simple year by year time series, and analyed using linear regression. It would be interesting to find a method of applying K-Medians to the area, to find the more dense locations, and to apply a support vector regression as well. Until then, enjoy!