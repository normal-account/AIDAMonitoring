from aida.aida import *;
import GPUtil
import sys
import numpy as np
import os.path
import pandas
import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor

host = 'yz_Server2'; dbname = 'sf01'; user = 'sf01'; passwd = 'sf01'; jobName = 'kmeans'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);
host = 'yz_Server2'
dbname='sf01'
user='sf01'
passwd='sf01'
jobName='xgboost'
port=55660

dw = AIDA.connect(host,dbname,user,passwd,jobName,port)

st_data = time.time()
new_lineitem_df= dw.lineitem.project(('l_quantity', 'l_extendedprice')).filter(Q('l_quantity',2,CMP.LT))

print(type(new_lineitem_df))
def process(self,data):
    data = copy.copy(data.rows);
    st_2 = time.time()
    #logging.info("copy data"+str(st_2-st_1))
    data_df = pd.DataFrame(data,columns=['l_quantity', 'l_extendedprice'])
    data_df = data_df.head(100000)
    norm_df = (data_df-data_df.mean())/data_df.std()
    x = torch.tensor(norm_df.values.astype(np.float32))
    x = x.view(x.shape[0],2)
    if x.size()[1] != 2:
        x = torch.transpose(x,0,1)
    en_2 = time.time()
    logging.info("process data"+str(en_2-st_2))
    self.x = x
dw._X(process,new_lineitem_df)

timeUsed = dw._KMeans(300, 200)
print(timeUsed)
