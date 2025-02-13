# coding: utf-8

# In this Python worflow we explore the Montreal Bixi biking data set for the year 2017 https://www.kaggle.com/aubertsigouin/biximtl/data
#
# We have additionally enriched this data set with the biking distance/duration available via Google map API as gmdata2017
#
# Our objective is to predict the "trip duration", given the distance between two stations.
#
# We are using the Pandas package with default settings for database connection to transfer data.

# Import required pacakges.

# In[1]:

import pandas.io.sql as psql;
import pandas as pd;
import pymonetdb.sql;
import ntpath;
import datetime;
from utils.timelog import TimeLog;
TL = TimeLog(__file__, '/tmp/tlog_{}_{}.txt'.format(ntpath.basename(__file__), datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')));

# Database connection information. We will use the default connection buffer settings etc for this workflow.

# In[2]:
TL.log('FEATURE_ENG');

host = 'cerberus';
dbname = 'bixi';
user = 'bixi';
passwd = 'bixi';

# Connect to the database.

# In[3]:

con = pymonetdb.Connection(dbname, hostname=host, username=user, password=passwd, autocommit=True);

# Let us find out what are the tables available in the database.
#
# Python DB API spec does not provide a mechanism to list the tables in the database, so it is left to the users to write a query depending on the RDBMS vendor.

# In[4]:

tblListSQL = "SELECT t.name as tableName "         "FROM sys.tables t "         "  INNER JOIN sys.schemas s "         "    ON t.schema_id = s.id "         "WHERE s.name = '{}'"        ";"

tables = psql.read_sql_query(sql=tblListSQL.format('bixi'), con=con);
print(tables);

# Let us take a peek into tripdata2017.
# For this purpose, we will create a pandas dataframe for tripdata2017 that we can reuse later.

# In[5]:

TL.log('DATA_LOAD');
tripdata2017 = pd.DataFrame(psql.read_sql_query('SELECT * FROM tripdata2017;', con));
TL.log('FEATURE_ENG');
print(tripdata2017.head());
print(tripdata2017.describe());

# So we have 4 million + records in tripdata2017. Also, the station codes are labels. We may have to enrich this information. Let us take a look at the contents of stations2017.

# In[6]:

TL.log('DATA_LOAD');
stations2017 = pd.DataFrame(psql.read_sql_query('SELECT * FROM stations2017;', con));
TL.log('FEATURE_ENG');
print(stations2017.head());
print(stations2017.describe());

# This is good, we have the longitude and latitude associated with each station, which can be used to enrich the tripdata.
#
# Since we have 546 stations, this gives the possibility of 546 x 546 = 298116 possible scenarios for trips. However, we need not be concerned with trips that started and ended at the same station as those are noise. Also, to weed out any further fluctuations in the input data set, we will limit ourselves to only those station combinations which has at the least 50 trips.
#
# For this purpose, we will use some relational-API like feature of Pandas.

# In[7]:

freqStations = tripdata2017.where(tripdata2017['stscode'] != tripdata2017['endscode']).dropna();
freqStations = pd.DataFrame({'numtrips': freqStations.groupby(['stscode', 'endscode']).size()}).reset_index();
freqStations = freqStations.where(freqStations['numtrips'] >= 50).dropna();
print(freqStations.head());
print(freqStations.describe());

# We can see that there are 19,300 station combinations that is of interest to us. Next we will include the longitude and latitude information of the start and end stations.

# In[8]:

freqStationsCord = pd.merge(freqStations, stations2017, left_on='stscode', right_on='scode').loc[:,
                   ['stscode', 'endscode', 'numtrips', 'slatitude', 'slongitude']].rename(index=str,
                                                                                          columns={'slatitude': 'stlat',
                                                                                                   'slongitude': 'stlong'});
freqStationsCord = pd.merge(freqStationsCord, stations2017, left_on='endscode', right_on='scode').loc[:,
                   ['stscode', 'endscode', 'numtrips', 'stlat', 'stlong', 'slatitude', 'slongitude']].rename(index=str,
                                                                                                             columns={
                                                                                                                 'slatitude': 'enlat',
                                                                                                                 'slongitude': 'enlong'});

print(freqStationsCord.head());

# It would be easier if we can translate the coordinates to a distance metric. Python's geopy module supports this computation using Vincenty's formula. This provides us with a distance as crow flies between two coordiantes. This might be a reasonable approximation of actual distance travelled in a trip.

# In[9]:

import geopy.distance;  # We will use this module to compute distance.


def computeDist(trip):
    # These are the inputs to Vincenty's formula.
    stlat = trip['stlat'];
    stlong = trip['stlong'];
    enlat = trip['enlat'];
    enlong = trip['enlong'];
    # populate the distance metric using longitude/latitude of coordinates.
    return int(geopy.distance.distance((stlat, stlong), (enlat, enlong)).meters);


freqStationsDist = pd.DataFrame(freqStationsCord);
freqStationsDist['vdistm'] = freqStationsDist.apply(computeDist, axis=1);
print(freqStationsDist.head());

# We can next enrich our trip data set with the distance information by joining these computed distances with each trip.

# In[10]:

tripData = pd.merge(tripdata2017, freqStationsDist, on=['stscode', 'endscode']).loc[:, ['id', 'duration', 'vdistm']];
print(tripData.head());
print(tripData.describe());

# So we have trip duration for each trip and the distance as crow flies, between the two stations involved in the trip.
#
# Also, we have about 2 million trips for which we have distance between stations metric. Given that there are only a few thousand unique values for distance, we might want to keep some values of distance apart for testing. For this purpose, we will first get distinct values for distance and then sort it.

# In[11]:

uniqueTripDist = tripData.loc[:, ['vdistm']].drop_duplicates().sort_values(by=['vdistm']);

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

trainTripDist = uniqueTripDist[~uniqueTripDist['vdistm'].isin(testTripDist['vdistm'])];
print(trainTripDist.head());
print(trainTripDist.tail());
print(trainTripDist.describe());

# Let us now extract the fields of interest to us for the training data, which is just the distance of each trip and it's duration

# In[14]:

trainData = tripData[tripData['vdistm'].isin(trainTripDist['vdistm'])];
trainData = trainData.loc[:, ['vdistm', 'duration']];
print(trainData.head());
print(trainData.tail());
print(trainData.describe());

# As the values are huge, we should normalize the data attributes. First get the max values for these attributes.

# In[15]:

maxdist = uniqueTripDist['vdistm'].max();
print(maxdist);
maxduration = tripData['duration'].max();
print(maxduration);

# Now let us normalize the training data. As we are working with integer data, we will also have to convert it to float. That can be accomplished by multiplying with 1.0.

# In[16]:

trainData['vdistm'] = trainData['vdistm'] / maxdist;
trainData['duration'] = trainData['duration'] / maxduration;

print(trainData.head());
print(trainData.tail());

# Our linear regression equation is of the form.
#
# dur = a + b*dist
#
# we will re-organize the training data set to fit this format and also setup our initial parameters for a and b.

# In[17]:

trainDataSet = trainData.loc[:, ['vdistm']];
trainDataSet.insert(0, 'x0', 1);
print(trainDataSet.head());
trainDataSetDuration = trainData.loc[:, ['duration']];
print(trainDataSetDuration.head());
# With pandas data frames, to do matrix multiplication, the column names should match.
params = pd.DataFrame({'x0': [1], 'vdistm': [1]}, index=['duration']);
print(params);

# Let us try to run a prediction using these parameters.

# In[18]:

pred = trainDataSet.dot(params.T);
print(pred.head());


# We need to compute the squared error for the predictions. Since we will be reusing them, we might as well store it as a function.

# In[19]:

def squaredErr(actual, predicted):
    return ((predicted - actual) ** 2).sum() / (2 * (actual.shape[0]));


# Let us see what is the error for the first iteration.

# In[20]:

sqerr = squaredErr(trainDataSetDuration, pred);
print(sqerr);


# We need to perform a gradient descent based on the squared errors. We will write another function to perform this.

# In[21]:

def gradDesc(actual, predicted, indata):
    return (predicted - actual).T.dot(indata) / actual.shape[0];


# Let us update our params using gradient descent using the error we got. We also need to use a learning rate, alpha (arbitrarily chosen).

# In[22]:

alpha = 0.1;

params = params - alpha * gradDesc(trainDataSetDuration, pred, trainDataSet);
print(params);

# Now let us try to use the updated params to train the model again and see if the error is decreasing.

# In[23]:

pred = trainDataSet.dot(params.T);
print(pred.head());
sqerr = squaredErr(trainDataSetDuration, pred);
print(sqerr);

# Before we proceed, may be we should check if google maps API's distance metric gives a better learning rate. Let us see what fields we can use from Google

# In[24]:

TL.log('DATA_LOAD');
gmdata2017 = pd.DataFrame(psql.read_sql_query('SELECT * FROM gmdata2017;', con));
TL.log('FEATURE_ENG');
print(gmdata2017.head());
print(gmdata2017.describe());

# We can build a new data set for the trips between frequently used station combination that includes google's distance.

# In[25]:

gtripData = pd.merge(gmdata2017, tripdata2017, on=['stscode', 'endscode']).loc[:,
            ['stscode', 'endscode', 'id', 'duration', 'gdistm', 'gduration']];
gtripData = pd.merge(gtripData, freqStations, on=['stscode', 'endscode']).loc[:,
            ['id', 'duration', 'gdistm', 'gduration']];
print(gtripData.head());
print(gtripData.describe());

# Google also provides its estimated duration for the trip. We will have to see in the end if our trained model is able to predict the trip duration better than google's estimate. So we will also save Google's estimate for the trip duration for that comparison.
#
# Next up, we need to format this dataset the same way we did the first one.

# In[26]:

guniqueTripDist = gtripData.loc[:, ['gdistm']].drop_duplicates().sort_values(by=['gdistm']);
gtestTripDist = guniqueTripDist[::3];

gtrainTripDist = guniqueTripDist[~guniqueTripDist['gdistm'].isin(gtestTripDist['gdistm'])];
gtrainData = gtripData[gtripData['gdistm'].isin(gtrainTripDist['gdistm'])];
gtrainData = gtrainData.loc[:, ['gdistm', 'duration']];

gmaxdist = guniqueTripDist['gdistm'].max();
print(gmaxdist);
gmaxduration = gtripData['duration'].max();
print(gmaxduration);
gtrainData['gdistm'] = gtrainData['gdistm'] / gmaxdist;
gtrainData['duration'] = gtrainData['duration'] / gmaxduration;

gtrainDataSet = gtrainData.loc[:, ['gdistm']];
gtrainDataSet.insert(0, 'x0', 1);
gtrainDataSetDuration = gtrainData.loc[:, ['duration']];
gparams = pd.DataFrame({'x0': [1], 'gdistm': [1]}, index=['duration']);

# Let us see how the error rate is progressing for the new dataset.

# In[27]:

gpred = gtrainDataSet.dot(gparams.T);
gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);
gparams = gparams - alpha * gradDesc(gtrainDataSetDuration, gpred, gtrainDataSet);
gpred = gtrainDataSet.dot(gparams.T);
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

for i in range(0, 1000):
    gpred = gtrainDataSet.dot(gparams.T);
    gparams = gparams - alpha * gradDesc(gtrainDataSetDuration, gpred, gtrainDataSet);
    if ((i + 1) % 100 == 0):
        print("Error rate after {} iterations is {}".format(i + 1, squaredErr(gtrainDataSetDuration, gpred)))

print(gparams);
gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);

# Let us see how our model performs in predictions against the test data set we had kept apart.

# In[29]:
TL.log('MODEL_TESTING');

gtestData = gtripData[gtripData['gdistm'].isin(gtestTripDist['gdistm'])];
gtestData = gtestData.loc[:, ['gdistm', 'duration', 'gduration']];
gtestData['gdistm'] = gtestData['gdistm'] / gmaxdist;
gtestData['duration'] = gtestData['duration'] / gmaxduration;
gtestDataSet = gtestData.loc[:, ['gdistm']];
gtestDataSet.insert(0, 'x0', 1);
gtestDataSetDuration = gtestData.loc[:, ['duration']];

gtestpred = gtestDataSet.dot(gparams.T);

gtestsqerr1 = squaredErr(gtestDataSetDuration * gmaxduration, gtestpred * gmaxduration);
print(gtestsqerr1);

# We would also like to check how the duration provided by Google maps' API hold up to the test data set.

# In[30]:

gtestsqerr2 = squaredErr(gtestDataSetDuration * gmaxduration,
                         gtestData.loc[:, ['gduration']].rename(columns={'gduration': 'duration'}));
print(gtestsqerr2);


# So yes, our model is able to do a better job.

# In[ ]:
TL.log('END');



