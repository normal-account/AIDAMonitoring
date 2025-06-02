
-- unlimited screen width, avoids truncation of columns in MonetDB's mclient
\w -1

SELECT 'FE_START_TIME', current_time();

-- See what tables we have in the database
\dt


-- Let us take a peek into tripdata2017.
SELECT * FROM tripdata2017 LIMIT 5;

-- We will look at the statistical distribution of certain fields of interest from the tripdata
SELECT
  COUNT(id) AS count_id
 ,COUNT(DISTINCT id) AS countd_id
 ,SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS countn_id
 ,MAX(id) AS max_id
 ,MIN(id) AS min_id
 ,AVG(id) AS avg_id
 ,SYS.MEDIAN(id) AS median_id
 ,SYS.QUANTILE(id, 0.25) AS q25_id
 ,SYS.QUANTILE(id, 0.50) AS q50_id
 ,SYS.QUANTILE(id, 0.75) AS q75_id
 ,SYS.STDDEV_POP(id) AS std_id
 ,COUNT(duration) AS count_duration
 ,COUNT(DISTINCT duration) AS countd_duration
 ,SUM(CASE WHEN duration IS NULL THEN 1 ELSE 0 END) AS countn_duration
 ,MAX(duration) AS max_duration
 ,MIN(duration) AS min_duration
 ,AVG(duration) AS avg_duration
 ,SYS.MEDIAN(duration) AS median_duration
 ,SYS.QUANTILE(duration, 0.25) AS q25_duration
 ,SYS.QUANTILE(duration, 0.50) AS q50_duration
 ,SYS.QUANTILE(duration, 0.75) AS q75_duration
 ,SYS.STDDEV_POP(duration) AS std_duration
FROM tripdata2017;


-- So we have 4 million + records in tripdata2017. Also, the station codes are labels. We may have to enrich this information.
-- Let us take a look at the contents of stations2017.
SELECT * FROM stations2017 LIMIT 5;

SELECT
  COUNT(scode) AS count_scode
 ,COUNT(DISTINCT scode) AS countd_scode
FROM stations2017;

--This is good, we have the longitude and latitude associated with each station, which can be used to enrich the tripdata.

--Since we have 546 stations, this gives the possibility of 546 x 546 = 298116 possible scenarios for trips.
-- However, we need not be concerned with trips that started and ended at the same station as those are noise.
-- Also, to weed out any further fluctuations in the input data set, we will limit ourselves to only those station combinations which has at the least 50 trips.
--

CREATE LOCAL TEMPORARY TABLE freqstations AS
  SELECT stscode, endscode, COUNT(*) as numtrips
  FROM tripdata2017
  WHERE stscode <> endscode
  GROUP BY stscode, endscode
  HAVING COUNT(*) >= 50
ON COMMIT PRESERVE ROWS
;

SELECT * FROM freqstations LIMIT 5;
SELECT COUNT(*) AS count_numtrips FROM freqstations;

--We can see that there are 19,300 station combinations that is of interest to us.
--Next stop, we need to include the longitude and latitude information of the start and end stations.
--

CREATE LOCAL TEMPORARY TABLE freqstationscord AS
  SELECT fs.*, sst.slatitude AS stlat, sst.slongitude AS stlong, est.slatitude AS enlat, est.slongitude AS enlong
  FROM freqstations fs, stations2017 sst, stations2017 est
  WHERE fs.stscode = sst.scode AND fs.endscode = est.scode
ON COMMIT PRESERVE ROWS
;
SELECT * FROM freqstationscord LIMIT 5;


--It would be easier if we can translate the coordinates to a distance metric.
-- Python's geopy module supports this computation using Vincenty's formula.
-- This provides us with a distance as crow flies between two coordiantes.
-- This might be a reasonable approximation of actual distance travelled in a trip.
-- We can use a UDF to accomplish this.

CREATE FUNCTION computevdist(stlat FLOAT, stlong FLOAT, enlat FLOAT, enlong FLOAT) RETURNS INTEGER
LANGUAGE PYTHON
{
    import numpy as np;
    import geopy.distance;  # We will use this module to compute distance.
    vdistm = np.empty(len(stlat), dtype=int);  # add a new empty column to hold distance.
    for i in range(0, len(stlat)):  # populate the distance metric using longitude/latitude of coordinates.
        vdistm[i] = int(geopy.distance.distance((stlat[i], stlong[i]), (enlat[i], enlong[i])).meters);
    return vdistm;
};
CREATE LOCAL TEMPORARY TABLE freqstationsdist AS
  SELECT stscode, endscode, numtrips, computevdist(stlat, stlong, enlat, enlong) AS vdistm
  FROM freqstationscord
ON COMMIT PRESERVE ROWS
;
SELECT * FROM freqstationsdist LIMIT 5;


--We can next enrich our trip data set with the distance information by joining these computed distances with each trip.


CREATE LOCAL TEMPORARY TABLE tripdata AS
  SELECT id, duration, vdistm
  FROM tripdata2017 td, freqstationsdist fs
  WHERE td.stscode = fs.stscode
    AND td.endscode = fs.endscode
ON COMMIT PRESERVE ROWS
;
SELECT * FROM tripdata LIMIT 5;


SELECT
  COUNT(duration) AS count_duration
 ,COUNT(DISTINCT duration) AS countd_duration
 ,SUM(CASE WHEN duration IS NULL THEN 1 ELSE 0 END) AS countn_duration
 ,MAX(duration) AS max_duration
 ,MIN(duration) AS min_duration
 ,AVG(duration) AS avg_duration
 ,SYS.MEDIAN(duration) AS median_duration
 ,SYS.QUANTILE(duration, 0.25) AS q25_duration
 ,SYS.QUANTILE(duration, 0.50) AS q50_duration
 ,SYS.QUANTILE(duration, 0.75) AS q75_duration
 ,SYS.STDDEV_POP(duration) AS std_duration
 ,COUNT(vdistm) AS count_vdistm
 ,COUNT(DISTINCT vdistm) AS countd_vdistm
 ,SUM(CASE WHEN vdistm IS NULL THEN 1 ELSE 0 END) AS countn_vdistm
 ,MAX(vdistm) AS max_vdistm
 ,MIN(vdistm) AS min_vdistm
 ,AVG(vdistm) AS avg_vdistm
 ,SYS.MEDIAN(vdistm) AS median_vdistm
 ,SYS.QUANTILE(vdistm, 0.25) AS q25_vdistm
 ,SYS.QUANTILE(vdistm, 0.50) AS q50_vdistm
 ,SYS.QUANTILE(vdistm, 0.75) AS q75_vdistm
 ,SYS.STDDEV_POP(vdistm) AS std_vdistm
FROM tripdata
;

--So we have trip duration for each trip and the distance as crow flies, between the two stations involved in the trip.
--
--Also, we have about 2 million trips for which we have distance between stations metric.
--Given that there are only a few thousand unique values for distance, we might want to keep some values of distance apart for testing.
--For this purpose, we will first get distinct values for distance and then create an ordering over it.

CREATE LOCAL TEMPORARY TABLE uniquetripdist AS
  SELECT vdistm, ROW_NUMBER() OVER(ORDER BY vdistm) AS rowidx
  FROM (SELECT DISTINCT vdistm FROM tripdata)x
ON COMMIT PRESERVE ROWS
;

SELECT
  COUNT(vdistm) AS count_vdistm
 ,COUNT(DISTINCT vdistm) AS countd_vdistm
 ,SUM(CASE WHEN vdistm IS NULL THEN 1 ELSE 0 END) AS countn_vdistm
 ,MAX(vdistm) AS max_vdistm
 ,MIN(vdistm) AS min_vdistm
 ,AVG(vdistm) AS avg_vdistm
 ,SYS.MEDIAN(vdistm) AS median_vdistm
 ,SYS.QUANTILE(vdistm, 0.25) AS q25_vdistm
 ,SYS.QUANTILE(vdistm, 0.50) AS q50_vdistm
 ,SYS.QUANTILE(vdistm, 0.75) AS q75_vdistm
 ,SYS.STDDEV_POP(vdistm) AS std_vdistm
FROM uniquetripdist
;
SELECT * FROM uniquetripdist WHERE rowidx <= 5;
SELECT * FROM uniquetripdist WHERE rowidx > 3652-5;

-- We will try to train a model based on this data set and see if it looks promising.
-- This will require a UDF.
CREATE FUNCTION BixiLinear()
RETURNS TABLE(sqerr1 FLOAT, sqerr2 FLOAT) LANGUAGE PYTHON
{
  import numpy as np;

  # Our linear regression equation is of the form.
  # dur = a + b*dist
  # We will normalize and extract the training data set to train this model.
  # For this purpose we will hardcode the max duration and max distance values we observed in the earlier output.
  # We are also keeping apart 1/3rd of the distance metrics for testing. Rest we will use to build the training data.
  trainDataSet_ = _conn.execute('SELECT 1 AS bias, CAST(1.0 AS FLOAT) * vdistm/{} AS vdistm , CAST(1.0 AS FLOAT) * duration/{} AS duration FROM ( SELECT vdistm, duration FROM tripdata WHERE vdistm IN ( SELECT vdistm FROM uniquetripdist WHERE NOT (rowidx%3 = 1) ) )x ;'.format(9074, 7199))

  trainDataSet  = np.stack( (trainDataSet_['bias'], trainDataSet_['vdistm']) );
  trainDataSetDuration = trainDataSet_['duration'];
  params = np.ones((2, 1));

  #Let us do a prediction on our training dataset.
  pred = params.T @ trainDataSet;

  # We need to compute the squared error for the predictions.
  def squaredErr(actual, predicted):
    return ((predicted - actual) ** 2).sum() / (2 * (actual.shape[0]));

  # Let us see what is the error for the first iteration.
  sqerr = squaredErr(trainDataSetDuration, pred);

  # We need to perform a gradient descent based on the squared errors. We will write another function to perform this.
  def gradDesc(actual, predicted, indata):
    return indata @ ((predicted - actual).T) / actual.shape[0];

  # Let us update our params using gradient descent using the error we got. We also need to use a learning rate, alpha (arbitrarily chosen).
  alpha = 0.1;
  params = params - alpha * gradDesc(trainDataSetDuration, pred, trainDataSet);

  # Now let us try to use the updated params to train the model again.
  pred = params.T @ trainDataSet;
  sqerr2 = squaredErr(trainDataSetDuration, pred);

  return {'sqerr1':sqerr, 'sqerr2':sqerr2 };
};
SELECT * FROM BixiLinear();


-- So the error rates are decreasing, so it might be a possible solution.
-- But before we proceed, may be we should check if google maps API's distance metric gives a better learning rate. Let us see what fields we can use from Google.

SELECT * FROM gmdata2017 LIMIT 5;
SELECT
  COUNT(gdistm) AS count_gdistm
 ,COUNT(DISTINCT gdistm) AS countd_gdistm
 ,SUM(CASE WHEN gdistm IS NULL THEN 1 ELSE 0 END) AS countn_gdistm
 ,MAX(gdistm) AS max_gdistm
 ,MIN(gdistm) AS min_gdistm
 ,AVG(gdistm) AS avg_gdistm
 ,SYS.MEDIAN(gdistm) AS median_gdistm
 ,SYS.QUANTILE(gdistm, 0.25) AS q25_gdistm
 ,SYS.QUANTILE(gdistm, 0.50) AS q50_gdistm
 ,SYS.QUANTILE(gdistm, 0.75) AS q75_gdistm
 ,SYS.STDDEV_POP(gdistm) AS std_gdistm
 ,COUNT(gduration) AS count_gduration
 ,COUNT(DISTINCT gduration) AS countd_gduration
 ,SUM(CASE WHEN gduration IS NULL THEN 1 ELSE 0 END) AS countn_gduration
 ,MAX(gduration) AS max_gduration
 ,MIN(gduration) AS min_gduration
 ,AVG(gduration) AS avg_gduration
 ,SYS.MEDIAN(gduration) AS median_gduration
 ,SYS.QUANTILE(gduration, 0.25) AS q25_gduration
 ,SYS.QUANTILE(gduration, 0.50) AS q50_gduration
 ,SYS.QUANTILE(gduration, 0.75) AS q75_gduration
 ,SYS.STDDEV_POP(gduration) AS std_gduration
FROM gmdata2017
;

-- We can build a new data set for the trips between frequently used station combination that includes google's distance.

CREATE LOCAL TEMPORARY TABLE gtripdata AS
  SELECT id, duration, gdistm, gduration
  FROM tripdata2017 td, freqstations fs, gmdata2017 gm
  WHERE td.stscode = fs.stscode
    AND td.endscode = fs.endscode
    AND td.stscode = gm.stscode
    AND td.endscode = gm.endscode
ON COMMIT PRESERVE ROWS
;
SELECT * FROM gtripdata LIMIT 5;
SELECT
  COUNT(gdistm) AS count_gdistm
 ,COUNT(DISTINCT gdistm) AS countd_gdistm
 ,SUM(CASE WHEN gdistm IS NULL THEN 1 ELSE 0 END) AS countn_gdistm
 ,MAX(gdistm) AS max_gdistm
 ,MIN(gdistm) AS min_gdistm
 ,AVG(gdistm) AS avg_gdistm
 ,SYS.MEDIAN(gdistm) AS median_gdistm
 ,SYS.QUANTILE(gdistm, 0.25) AS q25_gdistm
 ,SYS.QUANTILE(gdistm, 0.50) AS q50_gdistm
 ,SYS.QUANTILE(gdistm, 0.75) AS q75_gdistm
 ,SYS.STDDEV_POP(gdistm) AS std_gdistm
 ,COUNT(duration) AS count_duration
 ,COUNT(DISTINCT duration) AS countd_duration
 ,SUM(CASE WHEN duration IS NULL THEN 1 ELSE 0 END) AS countn_duration
 ,MAX(duration) AS max_duration
 ,MIN(duration) AS min_duration
 ,AVG(duration) AS avg_duration
 ,SYS.MEDIAN(duration) AS median_duration
 ,SYS.QUANTILE(duration, 0.25) AS q25_duration
 ,SYS.QUANTILE(duration, 0.50) AS q50_duration
 ,SYS.QUANTILE(duration, 0.75) AS q75_duration
 ,SYS.STDDEV_POP(duration) AS std_duration
 ,COUNT(gduration) AS count_gduration
 ,COUNT(DISTINCT gduration) AS countd_gduration
 ,SUM(CASE WHEN gduration IS NULL THEN 1 ELSE 0 END) AS countn_gduration
 ,MAX(gduration) AS max_gduration
 ,MIN(gduration) AS min_gduration
 ,AVG(gduration) AS avg_gduration
 ,SYS.MEDIAN(gduration) AS median_gduration
 ,SYS.QUANTILE(gduration, 0.25) AS q25_gduration
 ,SYS.QUANTILE(gduration, 0.50) AS q50_gduration
 ,SYS.QUANTILE(gduration, 0.75) AS q75_gduration
 ,SYS.STDDEV_POP(gduration) AS std_gduration
FROM gtripdata
;


-- Google also provides its estimated duration for the trip.
-- We will have to see in the end if our trained model is able to predict the trip duration better than google's estimate.
-- So we will also save Google's estimate for the trip duration for that comparison.
--
-- Next up, we need to format this dataset the same way we did the first one.

CREATE LOCAL TEMPORARY TABLE guniquetripdist AS
  SELECT gdistm, ROW_NUMBER() OVER(ORDER BY gdistm) AS rowidx
  FROM (SELECT DISTINCT gdistm FROM gtripdata)x
ON COMMIT PRESERVE ROWS
;

-- We will try to train a model based on this data set and see if it looks promising.
-- This will also be done via a UDF.
CREATE FUNCTION BixiLinearG()
RETURNS TABLE(sqerr1 FLOAT, sqerr2 FLOAT) LANGUAGE PYTHON
{
  import numpy as np;

  # We will normalize and extract the training data set to train this model.
  # For this purpose we will hardcode the max duration and max distance values we observed in the earlier output.
  # We are also keeping apart 1/3rd of the distance metrics for testing. Rest we will use to build the training data.
  gtrainDataSet_ = _conn.execute('SELECT 1 AS bias, CAST(1.0 AS FLOAT) * gdistm/{} AS gdistm , CAST(1.0 AS FLOAT) * duration/{} AS duration FROM ( SELECT gdistm, duration FROM gtripdata WHERE gdistm IN ( SELECT gdistm FROM guniquetripdist WHERE NOT (rowidx%3 = 1) ) )x ;'.format(14530, 7199))

  gtrainDataSet  = np.stack( (gtrainDataSet_['bias'], gtrainDataSet_['gdistm']) );
  gtrainDataSetDuration = gtrainDataSet_['duration'];
  gparams = np.ones((2, 1));

  #Let us do a prediction on our training dataset.
  gpred = gparams.T @ gtrainDataSet;

  # We need to compute the squared error for the predictions.
  def squaredErr(actual, predicted):
    return ((predicted - actual) ** 2).sum() / (2 * (actual.shape[0]));

  # Let us see what is the error for the first iteration.
  gsqerr = squaredErr(gtrainDataSetDuration, gpred);

  # We need to perform a gradient descent based on the squared errors.
  def gradDesc(actual, predicted, indata):
    return indata @ ((predicted - actual).T) / actual.shape[0];

  # Let us update our params using gradient descent using the error we got.
  alpha = 0.1;
  gparams = gparams - alpha * gradDesc(gtrainDataSetDuration, gpred, gtrainDataSet);

  # Now let us try to use the updated params to train the model again.
  gpred = gparams.T @ gtrainDataSet;
  gsqerr2 = squaredErr(gtrainDataSetDuration, gpred);

  return {'sqerr1':gsqerr, 'sqerr2':gsqerr2 };
};
SELECT * FROM BixiLinearG();


-- It looks like using Google maps' distance is giving us a slight advantage.
-- That makes sense, since Vincenty's formula computes distances as a crow flies,
-- where as Google maps' distance metric is based on the actual road network distances.
-- Better data gives better prediction results !
--
-- We are done with the feature selection and feature engineering phase for now.
--
-- Next we will proceed to train our linear regression model using the training data set.
--
-- We we need to re-write the original UDF a bit to add the iteration logic for linear regression.
DROP FUNCTION BixiLinearG;
CREATE FUNCTION BixiLinearG()
RETURNS TABLE
(
  sqerr1 FLOAT
, sqerr2 FLOAT
, timelog STRING
) LANGUAGE PYTHON
{
  import numpy as np;

  import time;
  fe_startt = time.time();

  # We will normalize and extract the training data set to train this model.
  # For this purpose we will hardcode the max duration and max distance values we observed in the earlier output.
  gmaxduration = 7199; gmaxdist = 14530;
  # We are also keeping apart 1/3rd of the distance metrics for testing. Rest we will use to build the training data.
  gtrainDataSet_ = _conn.execute('SELECT 1 AS bias, CAST(1.0 AS FLOAT) * gdistm/{} AS gdistm , CAST(1.0 AS FLOAT) * duration/{} AS duration FROM ( SELECT gdistm, duration FROM gtripdata WHERE gdistm IN ( SELECT gdistm FROM guniquetripdist WHERE NOT (rowidx%3 = 1) ) )x ;'.format(gmaxdist, gmaxduration))

  gtrainDataSet  = np.stack( (gtrainDataSet_['bias'], gtrainDataSet_['gdistm']) );
  gtrainDataSetDuration = gtrainDataSet_['duration'];
  gparams = np.ones((2, 1));

  # We need to compute the squared error for the predictions.
  def squaredErr(actual, predicted):
    return ((predicted - actual) ** 2).sum() / (2 * (actual.shape[0]));

  # We need to perform a gradient descent based on the squared errors.
  def gradDesc(actual, predicted, indata):
    return indata @ ((predicted - actual).T) / actual.shape[0];

  alpha = 0.1;
  mtr_startt = time.time();

  for i in range(0, 1000):
    gpred = gparams.T @ gtrainDataSet;
    gparams = gparams - alpha * gradDesc(gtrainDataSetDuration, gpred, gtrainDataSet);

  gsqerr = squaredErr(gtrainDataSetDuration, gpred);

  mte_startt = time.time();
  # Let us see how our model performs in predictions against the test data set we had kept apart.
  gtestDataSet_ =  _conn.execute('SELECT 1 AS bias, CAST(1.0 AS FLOAT) * gdistm/{} AS gdistm , CAST(1.0 AS FLOAT) * duration/{} AS duration , gduration FROM ( SELECT gdistm, duration, gduration FROM gtripData WHERE gdistm IN ( SELECT gdistm FROM guniqueTripDist WHERE (rowidx%3 = 1) ) )x ;'.format(gmaxdist, gmaxduration));
  gtestDataSet  = np.stack( (gtestDataSet_['bias'], gtestDataSet_['gdistm']) );
  gtestDataSetDuration = gtestDataSet_['duration'];

  gtestpred = gparams.T @ gtestDataSet;
  gtestsqerr1 = squaredErr(gtestDataSetDuration * gmaxduration, gtestpred * gmaxduration);

  # We would also like to check how the duration provided by Google maps API hold up to the test data set.
  gtestsqerr2 = squaredErr(gtestDataSetDuration * gmaxduration, gtestDataSet_['gduration']);

  end_t = time.time();
  timelogs = 'FEATURE_ENG={},MODEL_TRAINING={},MODEL_TESTING={}'.format(mtr_startt-fe_startt, mte_startt-mtr_startt, end_t-mte_startt);


  return {'sqerr1':gtestsqerr1, 'sqerr2':gtestsqerr2, 'timelog':timelogs };
};
SELECT 'FE_END_TIME', current_time();
SELECT * FROM BixiLinearG();


-- Clean ups.
DROP FUNCTION computevdist;
DROP FUNCTION BixiLinear;
DROP FUNCTION BixiLinearG;


