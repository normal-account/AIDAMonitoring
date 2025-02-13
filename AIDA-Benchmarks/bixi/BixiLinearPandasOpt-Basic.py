# coding: utf-8

# In this Python worflow we explore the Montreal Bixi biking data set for the year 2017 https://www.kaggle.com/aubertsigouin/biximtl/data
#
# We have additionally enriched this data set with the biking distance/duration available via Google map API as gmdata2017
#
# Our objective is to predict the "trip duration", given the distance between two stations.
#
# This is a "basic" workflow where the user directly builds the training dataset with minimal to no exploration, an unrealistic, but best case scenario.
#
# We are using the Pandas package with an explicitly optimized setting for database connection to transfer data.

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
databuffersize = 1000000;

# Connect to the database.

# In[3]:

con = pymonetdb.Connection(dbname, hostname=host, username=user, password=passwd, autocommit=True);
con.replysize = databuffersize;

# Use a SQL to read the minimal data required for training from the database.
#
# We will not be concerned with trips that started and ended at the same station as those are noise. Also, to weed out any further fluctuations in the input data set, we will limit ourselves to only those station combinations which has at the least 50 trips.

# In[4]:

TL.log('DATA_LOAD');
gtripData = pd.DataFrame(psql.read_sql_query(
    ' select t.duration, g.gdistm, g.gduration from (   select stscode, endscode   from bixi.tripdata2017   where stscode<>endscode   group by stscode, endscode   having count(*) >= 50 )s, tripdata2017 t, gmdata2017 g where t.stscode = s.stscode   and t.endscode = s.endscode   and t.stscode = g.stscode   and t.endscode = g.endscode ;',
    con));
TL.log('FEATURE_ENG');

# As there are multiple trips between the same stations, many trips will have the same distance. So we want to keep some values of distance apart for testing. For this purpose, we will first get distinct values for distance and then sort it.

# In[5]:

guniqueTripDist = gtripData.loc[:, ['gdistm']].drop_duplicates().sort_values(by=['gdistm']);

# We will keep roughly 30% of these distances apart for testing and the rest, we will use for training.

# In[6]:

gtestTripDist = guniqueTripDist[::3];
gtrainTripDist = guniqueTripDist[~guniqueTripDist['gdistm'].isin(gtestTripDist['gdistm'])];

# We will next extract the training data set and normalize its features.

# In[7]:

gtrainData = gtripData[gtripData['gdistm'].isin(gtrainTripDist['gdistm'])];
gtrainData = gtrainData.loc[:, ['gdistm', 'duration']];

gmaxdist = guniqueTripDist['gdistm'].max();
gmaxduration = gtripData['duration'].max();
gtrainData['gdistm'] = gtrainData['gdistm'] / gmaxdist;
gtrainData['duration'] = gtrainData['duration'] / gmaxduration;

# Our linear regression equation is of the form.
#
# dur = a + b*dist
#
# we will re-organize the training data set to fit this format and also setup our initial parameters for a and b.

# In[8]:

gtrainDataSet = gtrainData.loc[:, ['gdistm']];
gtrainDataSet.insert(0, 'x0', 1);
gtrainDataSetDuration = gtrainData.loc[:, ['duration']];
gparams = pd.DataFrame({'x0': [1], 'gdistm': [1]}, index=['duration']);

# Let us try to run a prediction using these parameters.

# In[9]:

gpred = gtrainDataSet.dot(gparams.T);


# We need to compute the squared error for the predictions. Since we will be reusing them, we might as well store it as a function.

# In[10]:

def squaredErr(actual, predicted):
    return ((predicted - actual) ** 2).sum() / (2 * (actual.shape[0]));


# Let us see what is the error for the first iteration.

# In[11]:

gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);


# We need to perform a gradient descent based on the squared errors. We will write another function to perform this.

# In[12]:

def gradDesc(actual, predicted, indata):
    return (predicted - actual).T.dot(indata) / actual.shape[0];


# Let us update our params using gradient descent using the error we got. We also need to use a learning rate, alpha (arbitrarily chosen).

# In[13]:

alpha = 0.1;

gparams = gparams - alpha * gradDesc(gtrainDataSetDuration, gpred, gtrainDataSet);
print(gparams);

# Now let us try to use the updated params to train the model again and see if the error is decreasing.

# In[14]:

gpred = gtrainDataSet.dot(gparams.T);
gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);

# This is good our error rate is decreasing with iteration. Hopefully this will help us construct the right parameters.
#
# We are done with the feature selection and feature engineering phase for now.
#
# Next we will proceed to train our linear regression model using the training data set.
#
# Meanwhile, we will also let it printout the error rate at frequent intervals so that we know it is decreasing.

# In[15]:
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

# In[16]:
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

# In[17]:

gtestsqerr2 = squaredErr(gtestDataSetDuration * gmaxduration,
                         gtestData.loc[:, ['gduration']].rename(columns={'gduration': 'duration'}));
print(gtestsqerr2);


# So yes, our model is able to do a better job.

# In[ ]:
TL.log('END');



