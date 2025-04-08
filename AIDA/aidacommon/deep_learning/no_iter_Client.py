from aida.aida import *;
import sys;
import pandas as pd;
import numpy as np;
import time as time;
from ctypes import *
import os
import torch;
import psutil;
import torch.nn as nn;
import collections;
from aidas.models import *

host = 'yz_Server2'; dbname = 'sf01'; user = 'sf01'; passwd = 'sf01'; jobName = 'torchLinear'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);

#n = 100
#get_data_start = time.time()
#df = pd.DataFrame(np.random.randn(n))
#df.columns = ['A']
#df['B'] = np.random.randn(n)
#df['C'] = np.random.randn(n)
#df['D'] = np.random.randn(n)
#df['E'] = np.random.randn(n)
#df['Y'] = 5 + 3 * df.A + 6 * df.B ** 2 + 7 * df.C ** 3 + 2 * df.D ** 2 + 8 * df.E * df.D + np.random.randn(n)
new_aapl_df= dw.aapl.project(('open', 'high', 'low','close'))

#dataset = df.copy()
#dw.dataset = dataset

"""
def process(self, dataset):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("Y")
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop('Y')
    test_labels = test_dataset.pop('Y')

    #train_target = torch.tensor(train_labels.values.astype(np.float32))
    #train_target = train_target.view(train_target.shape[0], 1)
    #test_target = torch.tensor(test_labels.values.astype(np.float32))
    #test_target = test_target.view(test_target.shape[0], 1)

    #become a col of np.float32
    train_target =train_labels.values.astype(np.float32)
    train_target = train_target.reshape(train_target.shape[0], 1)
    test_target = test_labels.values.astype(np.float32)
    test_target = test_target.reshape(test_target.shape[0], 1)
    train_target = torch.tensor(train_target)
    test_target = torch.tensor(test_target)

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    normed_train_data = torch.tensor(normed_train_data.values.astype(np.float32))
    normed_test_data = torch.tensor(normed_test_data.values.astype(np.float32))
    inFeatures = len(train_dataset.keys())

    self.x_train = normed_train_data
    self.y_train = train_target
    self.x_test = normed_test_data
    self.y_test = test_target
    self.inFeatures = normed_train_data.shape
    
    
"""
def process(self,data):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    data = copy.copy(data.rows);
    data_df = pd.DataFrame(data,columns=['open', 'high', 'low','close'])
    dataset = data_df.copy()
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("close")
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop('close')
    test_labels = test_dataset.pop('close')


    #become a col of np.float32
    train_target =train_labels.values.astype(np.float32)
    train_target = train_target.reshape(train_target.shape[0], 1)
    test_target = test_labels.values.astype(np.float32)
    test_target = test_target.reshape(test_target.shape[0], 1)

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)


    normed_train_data =normed_train_data.values.astype(np.float32)
    normed_test_data = normed_test_data.values.astype(np.float32)

    self.x_train = torch.tensor(normed_train_data)
    self.y_train = torch.tensor(train_target)
    self.x_test = torch.tensor(normed_test_data)
    self.y_test = torch.tensor(test_target)
    self.inFeatures = len(train_dataset.keys())

"""
def get_training_model(inFeatures, hiddenDim=16, nbClasses=1):
        # construct a shallow, sequential neural network
    model = nn.Sequential(collections.OrderedDict([
        ("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
        ("activation_1", nn.ReLU()),
        ("hidden_layer_2", nn.Linear(hiddenDim, hiddenDim)),
        ("activation_2", nn.ReLU()),
        ("output_layer", nn.Linear(hiddenDim, nbClasses))
    ]))
        # return the sequential model
    return model
"""
def __init__(self, inFeatures, hiddenDim=16, nbClasses=1):
    super(Model, self).__init__()
    self.inFeatures = inFeatures
    self.hidden_layer_1 = nn.Linear(inFeatures, hiddenDim)
    self.activation_1 = nn.ReLU()
    self.hidden_layer_2 = nn.Linear(hiddenDim, hiddenDim)
    self.activation_2 = nn.ReLU()
    self.output_layer = nn.Linear(hiddenDim, nbClasses)

def forward(self, x):
    x = self.hidden_layer_1(x)
    x = self.activation_1(x)
    x = self.hidden_layer_2(x)
    x = self.activation_2(x)
    x = self.output_layer(x)
    # print(f'shape after all dense layers: {x.shape}')
    return x

dw._Preprocess(process, new_aapl_df)
Model.__init__ = __init__
Model.forward = forward
model = Model(inFeatures=dw.inFeatures)
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
using_gpu = True
return_value = dw._NN(epochs=10000000000, model=model, forward=forward, criterion=criterion, optimizer=optimizer, name="torchLinear", loss_limit=2)
print(f"time taken: {return_value}")
