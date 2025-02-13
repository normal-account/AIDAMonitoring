from aida.aida import *
import pandas as pd
import numpy as np

host = 'localhost'
dbname = 'bixi'
user = 'bixi'
passwd = 'bixi'
jobName = 'job1'
port = 55660

dw = AIDA.connect(host,dbname,user,passwd,jobName,port)

print(type(dw))
print(dw._getBufferHitRate())

#data1 = dw.pg_locks
#data1 = dw.tripdata2017


#print(type(data1))

#print(data1.head())

# training_set = dw.tripdata2017.join(dw.gmdata2017, ('stscode', 'endscode'), ('stscode', 'endscode')
#                                 , ('id', 'duration', 'ismember', 'stscode', 'endscode'), ('gdistm', 'gduration'))

# # for regression test, we use the gdistm and gduration to predict the actual duration of a trip
# regression_x = training_set[:, ['gdistm', 'gduration']]
# regression_y = training_set[:, ['duration']]

# regression_predict = np.array([[580, 867], [90, 160], [3050, 2256]])
# regression_rs = pd.DataFrame(data=regression_predict, columns=['gdistm', 'gduration'])

# # for linear regression, we use the start/end station and the duration to predict if the user is a member
# tree_x = training_set[:, ['stscode', 'endscode', 'duration']]
# tree_y = training_set[:, ['ismember']]

# tree_predict = [[6154,6148, 100], [6148, 6154, 427], [6062, 6062, 605]]
# tree_rs = pd.DataFrame(data=tree_predict, columns=['stscode', 'endscode', 'duration'])

# # train and eval with linear regression
# model = dw._linearRegression()
# model.fit(regression_x, regression_y)
# rs = model.predict(regression_predict)
# regression_rs['linear_y'] = rs

# # train and eval with logistic regression
# model = dw._logisticRegression()
# model.fit(regression_x, regression_y)
# rs = model.predict(regression_predict)
# regression_rs['logistic_y'] = rs

# print(regression_rs)

# # train and eval with decision tree
# model = dw._decisionTree()
# model.fit(tree_x, tree_y)
# tree_rs['tree_y'] = model.predict(tree_predict)

# # view the input and the predicted value
# print(tree_rs)


dw._close()

    



