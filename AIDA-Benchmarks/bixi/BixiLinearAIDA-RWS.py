
# coding: utf-8

# In this Python worflow we explore the Montreal Bixi biking data set for the year 2017 https://www.kaggle.com/aubertsigouin/biximtl/data
#
# We have additionally enriched this data set with the biking distance/duration available via Google map API as gmdata2017
#
# Our objective is to predict the "trip duration", given the distance between two stations.

# Import AIDA components

# In[1]:

from aida.aida import *;
import ntpath;
import datetime;
from utils.timelog import TimeLog;
TL = TimeLog(__file__, '/tmp/tlog_{}_{}.txt'.format(ntpath.basename(__file__), datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')));


# Connection information to AIDA's server

# In[2]:
TL.log('FEATURE_ENG');

host='cerberus'; dbname='bixi'; user='bixi'; passwd='bixi'; jobName='bixiLinear'; port=55660;


# Establish a connection and get a handle to the database workspace.

# In[3]:

dw = AIDA.connect(host, dbname, user, passwd, jobName, port);


# Let us see what tables we have in the database

# In[4]:

print(dw._tables());


# Let us take a peek into tripdata2017.
#
# We can see the attributes and explore some sample data.This can be accomplished via the head() or tail() functions (similar to Pandas API) provided by TabularData that sends a sample of data from the server to the client side.
#
# Further we can also use the describe() function to look at the data distribution characteristics. This semantic is very similar to the functionality provided by pandas DataFrame to get a summary of the overall distribution of each attribute.

# In[5]:

print(dw.tripdata2017.head());
print(dw.tripdata2017.describe());


# So we have 4 million + records in tripdata2017. Also, the station codes are labels. We may have to enrich this information.
# Let us take a look at the contents of stations2017.

# In[6]:

print(dw.stations2017.head());
print(dw.stations2017.describe());


# This is good, we have the longitude and latitude associated with each station, which can be used to enrich the tripdata.

# Since we have 546 stations, this gives the possibility of 546 x 546 = 298116 possible scenarios for trips. However, we need not be concerned with trips that started and ended at the same station as those are noise. Also, to weed out any further fluctuations in the input data set, we will limit ourselves to only those station combinations which has at the least 50 trips.
#
# We can use AIDA's powerful relational API to accomplish this.

# In[7]:

freqStations = dw.tripdata2017.filter(Q('stscode', 'endscode', CMP.NE))     .aggregate(('stscode','endscode',{COUNT('*'):'numtrips'}), ('stscode','endscode'))     .filter(Q('numtrips',C(50), CMP.GTE));
print(freqStations.head());
print(freqStations.describe());


# We can see that there are 19,300 station combinations that is of interest to us.
# Next stop, we need to include the longitude and latitude information of the start and end stations.
#
# This can be done by joining with the station information using AIDA's relational join operator.

# In[8]:

freqStationsCord = freqStations     .join(dw.stations2017, ('stscode',), ('scode',), COL.ALL, ({'slatitude':'stlat'}, {'slongitude':'stlong'}))     .join(dw.stations2017, ('endscode',), ('scode',), COL.ALL, ({'slatitude':'enlat'}, {'slongitude':'enlong'}));
print(freqStationsCord.head());


# It would be easier if we can translate the coordinates to a distance metric. Python's geopy module supports this computation using Vincenty's formula. This provides us with a distance as crow flies between two coordiantes. This might be a reasonable approximation of actual distance travelled in a trip.
#
# Using TabularData's user transform operator, we can generate a dataset which also includes this distance metric.

# In[9]:

def computeDist(tblrData):
    import geopy.distance;     #We will use this module to compute distance.
    import copy, numpy as np;
    #We are going to keep all the columns of the source tabularData object.
    data = copy.copy(tblrData.rows); #This only makes a copy of the metadata, but retains original column data
    vdistm = data['vdistm'] = np.empty(tblrData.numRows, dtype=int); #add a new empty column to hold distance.
    #These are the inputs to Vincenty's formula.
    stlat = data['stlat']; stlong = data['stlong']; enlat = data['enlat']; enlong = data['enlong'];
    for i in range(0, tblrData.numRows): #populate the distance metric using longitude/latitude of coordinates.
        vdistm[i] = int(geopy.distance.distance((stlat[i],stlong[i]), (enlat[i],enlong[i])).meters);
    return data;

freqStationsDist = freqStationsCord._U(computeDist); #Execute the user transform
print(freqStationsDist.head());                      #Take a peek at a sample data.


# We can next enrich our trip data set with the distance information by joining these computed distances with each trip.

# In[10]:

tripData = dw.tripdata2017.join(freqStationsDist, ('stscode','endscode'), ('stscode', 'endscode')
                                             , ('id', 'duration'), ('vdistm',));
print(tripData.head());
print(tripData.describe());


# So we have trip duration for each trip and the distance as crow flies, between the two stations involved in the trip.
#
# Also, we have about 2 million trips for which we have distance between stations metric.
# Given that there are only a few thousand unique values for distance, we might want to keep some values of distance apart for testing.
# For this purpose, we will first get distinct values for distance and then sort it.

# In[11]:

uniqueTripDist = tripData[:,['vdistm']].distinct().order('vdistm');
print(uniqueTripDist.head());
print(uniqueTripDist.tail());
print(uniqueTripDist.describe());


# We will keep some data apart for testing. A rule of thumb is 30%. The neat trick below sets apart 33%, across the entire range of distance values. close enough.

# In[12]:

testTripDist = uniqueTripDist[::3];
print(testTripDist.head());
print(testTripDist.tail());
print(testTripDist.describe());


# Now let us get the remaining values for distances to be used for training.

# In[13]:

trainTripDist = uniqueTripDist.filter(Q('vdistm', testTripDist, CMP.NOTIN));
print(trainTripDist.head());
print(trainTripDist.tail());
print(trainTripDist.describe());


# Let us now extract the fields of interest to us for the training data, which is just the distance of each trip and it's duration

# In[14]:

trainData = tripData.project(('vdistm', 'duration')).filter(Q('vdistm', trainTripDist, CMP.IN));
print(trainData.head());
print(trainData.tail());
print(trainData.describe());


# As the values are huge, we should normalize the data attributes. First get the max values for these attributes.

# In[15]:

maxdist = uniqueTripDist.max('vdistm');
print(maxdist);
maxduration = tripData.max('duration');
print(maxduration);


# Now let us normalize the training data. As we are working with integer data, we will also have to convert it to float. That can be accomplished by multiplying with 1.0.

# In[16]:

trainData = trainData.project((1.0*F('vdistm')/maxdist, 1.0*F('duration')/maxduration));

print(trainData.head());
print(trainData.tail());


# Our linear regression equation is of the form.
#
# dur = a + b*dist
#
# we will re-organize the training data set to fit this format and also setup our initial parameters for a and b.

# In[17]:

trainDataSet = dw._ones((trainData.numRows, 1), ("x0",)).hstack(trainData[:,['vdistm']]);
print(trainDataSet.head());
trainDataSetDuration = trainData[:,['duration']];
print(trainDataSetDuration.head());
params = dw._ones((1,2), ("a","b"));
print(params.rows);


# Let us try to run a prediction using these parameters.

# In[18]:

pred = trainDataSet @ params.T;
print(pred.columns);
print(pred.head());


# We need to compute the squared error for the predictions. Since we will be reusing them, we might as well store it as a function.

# In[19]:

def squaredErr(actual, predicted):
    return ((predicted-actual)**2).sum()/(2*(actual.shape[0]));


# Let us see what is the error for the first iteration.

# In[20]:

sqerr = squaredErr(trainDataSetDuration, pred);
print(sqerr);


# We need to perform a gradient descent based on the squared errors. We will write another function to perform this.

# In[21]:

def gradDesc(actual, predicted, indata):
    return (predicted-actual).T @ indata / actual.shape[0];


# Let us update our params using gradient descent using the error we got. We also need to use a learning rate, alpha (arbitrarily chosen).

# In[22]:

alpha = 0.1;

params = params - alpha * gradDesc(trainDataSetDuration, pred, trainDataSet);
print(params.rows);


# Now let us try to use the updated params to train the model again and see if the error is decreasing.

# In[23]:

pred = trainDataSet @ params.T;
print(pred.head());
sqerr = squaredErr(trainDataSetDuration, pred);
print(sqerr);


# Before we proceed, may be we should check if google maps API's distance metric gives a better learning rate. Let us see what fields we can use from Google.

# In[24]:

print(dw.gmdata2017.head());
print(dw.gmdata2017.describe());


# We can build a new data set for the trips between frequently used station combination that includes google's distance.

# In[25]:

gtripData = dw.gmdata2017     .join(dw.tripdata2017, ('stscode','endscode'), ('stscode', 'endscode'), COL.ALL, COL.ALL)     .join(freqStations, ('stscode','endscode'), ('stscode', 'endscode')                           , ('id', 'duration', 'gdistm', 'gduration') );
print(gtripData.head());
print(gtripData.describe());


# Google also provides its estimated duration for the trip. We will have to see in the end if our trained model is able to predict the trip duration better than google's estimate. So we will also save Google's estimate for the trip duration for that comparison.

# Next up, we need to format this dataset the same way we did the first one.

# In[26]:

guniqueTripDist = gtripData[:,['gdistm']].distinct().order('gdistm');
gtestTripDist = guniqueTripDist[::3];
gtrainTripDist = guniqueTripDist.filter(Q('gdistm', gtestTripDist, CMP.NOTIN));
gtrainData = gtripData.project(('gdistm', 'duration')).filter(Q('gdistm', gtrainTripDist, CMP.IN));

gmaxdist = guniqueTripDist.max('gdistm');
print(gmaxdist);
gmaxduration = gtripData.max('duration');
print(gmaxduration);
gtrainData = gtrainData.project((1.0*F('gdistm')/gmaxdist, 1.0*F('duration')/gmaxduration));

gtrainDataSet = dw._ones((gtrainData.numRows, 1), ("x0",)).hstack(gtrainData[:,['gdistm']]);
gtrainDataSetDuration = gtrainData[:,['duration']];
gparams = dw._ones((1,2), ("a","b"));


# Let us see how the error rate is progressing for the new dataset.

# In[27]:

gpred = gtrainDataSet @ gparams.T;
gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);
gparams = gparams - alpha * gradDesc(gtrainDataSetDuration, gpred, gtrainDataSet);
gpred = gtrainDataSet @ gparams.T;
gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);


# It looks like using Google maps' distance is giving us a slight advantage. That makes sense, since Vincenty's formula computes distances as a crow flies, where as Google maps' distance metric is based on the actual road network distances. Better data gives better prediction results !
#
# We are done with the feature selection and feature engineering phase for now.
#
# Next we will proceed to train our linear regression model using the training data set.
#
# Meanwhile, we will also let it printout the error rate at frequent intervals so that we know it is decreasing.

# In[28]:
TL.log('MODEL_TRAINING');

def trainModel(w, numiters, alpha):
    gtrainDataSet = w.gtrainDataSet; gtrainDataSetDuration=w.gtrainDataSetDuration;
    gparams = w.gparams; gradDesc = w.gradDesc;
    for i in range(0, numiters):
        gpred = gtrainDataSet @ gparams.T;
        gparams = gparams - alpha*gradDesc(gtrainDataSetDuration, gpred, gtrainDataSet);
    w.gpred = gpred; w.gparams = gparams;

#Export any objects and functions required for execution in the remote workspace.
dw.gtrainDataSet = gtrainDataSet; dw.gtrainDataSetDuration = gtrainDataSetDuration;
dw.gparams = gparams; dw.gradDesc = gradDesc;

for i in range(0, 10):
    dw._X(trainModel, 100, alpha);
    print("Error rate after {} iterations is {}".format((i+1)*100, squaredErr(gtrainDataSetDuration, dw.gpred)))

gparams = dw.gparams; gpred = dw.gpred;
print(gparams.rows);
gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);


# Let us see how our model performs in predictions against the test data set we had kept apart.

# In[29]:
TL.log('MODEL_TESTING');

gtestData = gtripData.project(('gdistm', 'duration', 'gduration')).filter(Q('gdistm', gtestTripDist, CMP.IN));
gtestData = gtestData.project((1.0*F('gdistm')/gmaxdist, 1.0*F('duration')/gmaxduration, 'gduration'));
gtestDataSet = dw._ones((gtestData.numRows, 1), ("x0",)).hstack(gtestData[:,['gdistm']]);
gtestDataSetDuration = gtestData[:,['duration']];

gtestpred = gtestDataSet @ gparams.T;

gtestsqerr1 = squaredErr(gtestDataSetDuration*gmaxduration, gtestpred*gmaxduration);
print(gtestsqerr1);


# We would also like to check how the duration provided by Google maps' API hold up to the test data set.

# In[30]:

gtestsqerr2 = squaredErr(gtestDataSetDuration*gmaxduration, gtestData[:,['gduration']]);
print(gtestsqerr2);


# So yes, our model is able to do a better job.

# In[ ]:
TL.log('END');



