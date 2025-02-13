
# coding: utf-8

# In this Python worflow we explore the Montreal Bixi biking data set for the year 2017 https://www.kaggle.com/aubertsigouin/biximtl/data
#
# We have additionally enriched this data set with the biking distance/duration available via Google map API as gmdata2017
#
# Our objective is to predict the "trip duration", given the distance between two stations.
#
# This is a "basic" workflow where the user directly builds the training dataset with minimal to no exploration, an unrealistic, but best case scenario.

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


# We will not be concerned with trips that started and ended at the same station as those are noise. Also, to weed out any further fluctuations in the input data set, we will limit ourselves to only those station combinations which has at the least 50 trips.
#
# We can use AIDA's powerful relational API to accomplish this.

# In[4]:

freqStations = dw.tripdata2017.filter(Q('stscode', 'endscode', CMP.NE))     .aggregate(('stscode','endscode',{COUNT('*'):'numtrips'}), ('stscode','endscode'))     .filter(Q('numtrips',C(50), CMP.GTE));


# Next we will enrich the trip data set by using the distance information provided by the Google maps' API.
# This can be accomplished by the relational join operators provided by AIDA.

# Google also provides its estimated duration for the trip. We will have to see in the end if our trained model is able to predict the trip duration better than google's estimate. So we will also save Google's estimate for the trip duration for that comparison.

# In[5]:

gtripData = dw.gmdata2017     .join(dw.tripdata2017, ('stscode','endscode'), ('stscode', 'endscode'), COL.ALL, COL.ALL)     .join(freqStations, ('stscode','endscode'), ('stscode', 'endscode')                           , ('id', 'duration', 'gdistm', 'gduration') );


# As there are multiple trips between the same stations, many trips will have the same distance. So we want to keep some values of distance apart for testing. For this purpose, we will first get distinct values for distance and then sort it.

# In[6]:

guniqueTripDist = gtripData[:,['gdistm']].distinct().order('gdistm');


# We will keep roughly 30% of these distances apart for testing and the rest, we will use for training.

# In[7]:

gtestTripDist = guniqueTripDist[::3];
gtrainTripDist = guniqueTripDist.filter(Q('gdistm', gtestTripDist, CMP.NOTIN));


# We will next extract the training data set and normalize its features.

# In[8]:

gtrainData = gtripData.project(('gdistm', 'duration')).filter(Q('gdistm', gtrainTripDist, CMP.IN));

gmaxdist = guniqueTripDist.max('gdistm');
gmaxduration = gtripData.max('duration');
gtrainData = gtrainData.project((1.0*F('gdistm')/gmaxdist, 1.0*F('duration')/gmaxduration));


# Our linear regression equation is of the form.
#
# dur = a + b*dist
#
# we will re-organize the training data set to fit this format and also setup our initial parameters for a and b.

# In[9]:

gtrainDataSet = dw._ones((gtrainData.numRows, 1), ("x0",)).hstack(gtrainData[:,['gdistm']]);
gtrainDataSetDuration = gtrainData[:,['duration']];
gparams = dw._ones((1,2), ("a","b"));


# Let us try to run a prediction using these parameters.

# In[10]:

gpred = gtrainDataSet @ gparams.T;


# We need to compute the squared error for the predictions. Since we will be reusing them, we might as well store it as a function.

# In[11]:

def squaredErr(actual, predicted):
    return ((predicted-actual)**2).sum()/(2*(actual.shape[0]));


# Let us see what is the error for the first iteration.

# In[12]:

gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);


# We need to perform a gradient descent based on the squared errors. We will write another function to perform this.

# In[13]:

def gradDesc(actual, predicted, indata):
    return (predicted-actual).T @ indata / actual.shape[0];


# Let us update our params using gradient descent using the error we got. We also need to use a learning rate, alpha (arbitrarily chosen).

# In[14]:

alpha = 0.1;

gparams = gparams - alpha * gradDesc(gtrainDataSetDuration, gpred, gtrainDataSet);
print(gparams.rows);


# Now let us try to use the updated params to train the model again and see if the error is decreasing.

# In[15]:

gpred = gtrainDataSet @ gparams.T;
gsqerr = squaredErr(gtrainDataSetDuration, gpred);
print(gsqerr);


# This is good our error rate is decreasing with iteration. Hopefully this will help us construct the right parameters.
#
# We are done with the feature selection and feature engineering phase for now.
#
# Next we will proceed to train our linear regression model using the training data set.
#
# Meanwhile, we will also let it printout the error rate at frequent intervals so that we know it is decreasing.

# In[16]:
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

# In[17]:
TL.log('MODEL_TESTING');

gtestData = gtripData.project(('gdistm', 'duration', 'gduration')).filter(Q('gdistm', gtestTripDist, CMP.IN));
gtestData = gtestData.project((1.0*F('gdistm')/gmaxdist, 1.0*F('duration')/gmaxduration, 'gduration'));
gtestDataSet = dw._ones((gtestData.numRows, 1), ("x0",)).hstack(gtestData[:,['gdistm']]);
gtestDataSetDuration = gtestData[:,['duration']];

gtestpred = gtestDataSet @ gparams.T;

gtestsqerr1 = squaredErr(gtestDataSetDuration*gmaxduration, gtestpred*gmaxduration);
print(gtestsqerr1);


# We would also like to check how the duration provided by Google maps' API hold up to the test data set.

# In[18]:

gtestsqerr2 = squaredErr(gtestDataSetDuration*gmaxduration, gtestData[:,['gduration']]);
print(gtestsqerr2);


# So yes, our model is able to do a better job.

# In[ ]:
TL.log('END');



