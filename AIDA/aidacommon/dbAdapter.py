import weakref;
import inspect;
import copyreg;
from enum import Enum;
from abc import ABCMeta, abstractmethod;

import collections;

import logging;

import numpy as np;
import sys
#import cupy as cp;
import time
import GPUtil
import psutil
import threading
import multiprocessing
from multiprocessing import Process


import random;
from io import BytesIO;
from PIL import Image;

from aidacommon.aidaConfig import AConfig;
from aidacommon.rop import ROMgr;
from aidacommon.rdborm import *;
from aidas.scheduler import ScheduleManager as scheduler;
from aidas.models import *;

from aidacommon.gbackend import GBackendApp;

import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import ast

import torch
from sklearn import datasets
import sys
import torch.nn as nn
import torch.nn.functional as F
import GPUtil



# helper class and methods that convert TabularData Object to numpy arrays
class DataConversion:

    # a static function to identify whether a variable contains a numerical value
    def is_number(n):
        try:
            float(n)
        except ValueError:
            return False
        return True

    # extract_X and extract_y using .matrix instead of .cdata
    #def extract_X(TabularDataX):
    #   data_X=TabularDataX.matrix
    #   row_indices=list()

    #   for i in range(data_X.shape[0]):
    #       if DataConversion.is_number(data_X[i][0]):
    #           row_indices.append(i)

    #   X=data_X[row_indices,].transpose()
    #   return X

    #def extract_y(TabularDatay):
    #   data_y=TabularDatay.matrix
    #   i=0
    #   while (i<data_y.shape[0]):
    #        if DataConversion.is_number(data_y[i][0]):
    #           break
    #       else:
    #            i+=1
    #   y=data_y[i]
    #    return y
    
    # a static function to convert the TabularData containing X values into numpy matrix 
    def extract_X(TabularDataX):
        data_X=TabularDataX.cdata
        key_list=list()
        for key in data_X:
            key_list.append(key)

        # a list of indices of keys which contain numerical values in TabularDataX
        numerical_indices=list()
        for i in range(len(key_list)):
            # assume all values under one column are of the same type
            # if the first value of a column is numerical, then the column is a numerical column
            n=data_X.get(key_list[i])[0]
            if DataConversion.is_number(n):
                numerical_indices.append(i)
        # if TabularDataX does not have numerical columns
        if (len(numerical_indices)==0):
            raise ValueError("Error: No X values are numerical")

        # TabularDataX has numerical columns, then extract the numpy arrays as features 
        X=data_X.get(key_list[numerical_indices[0]]).reshape(-1,1)
        for index in range(1,len(numerical_indices)):
            # a matrix of shape (n_sample,n_feature)
            X=np.concatenate((X,data_X.get(key_list[numerical_indices[index]]).reshape(-1,1)),axis=1)

        return X

    # a static function to convert the TabularData containing y values into numpy array
    def extract_y(TabularDatay):
        data_y=TabularDatay.cdata
        count=0
        for key in data_y:
            n=data_y[key][0]
            if DataConversion.is_number(n):
                break
            else:
                count+=1

        # if TabularDataObject2 does not have numerical columns
        if count>=len(data_y):
            raise ValueError("Error: No y values are numerical")

        # TabularDataObject2 has numerical columns, then extract the numpy aray as label
        y=data_y.get(key)
        return y

class LogisticRegressionModel:
    
    # initialize a LogisticRegressionModel object with "model" attribute containing an actual LogisticRegression object from the sklearn module    
    def __init__(self,*args,**kwargs):
        self.model=LogisticRegression(*args,**kwargs)

    # a function that returns the actual LinearRegression object which the called LogisticRegressionModel object wraps around
    def get_model(self):
        return self.model

    def decision_function(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.decision_function(X)

    def fit(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        self.model.fit(X,y,sample_weight)
        return self

    def predict(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict(X)

    def predict_log_proba(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict_log_proba(X)

    def predict_proba(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict_proba(X)

    def score(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        return self.model.score(X,y,sample_weight)

    def __getattribute__(self,item):
        # if the called function/attribute does not require X,y tabularData conversion, get the attribute value by calling the function on the actual LogisticRegression model in skLearn module

        # check if this object has the requested attribute
        try:
            return super().__getattribute__(item)
        except:
            pass;
        # otherwise fetch it from the actual linear regression object
        return getattr(self.model,item)

copyreg.pickle(LogisticRegressionModel,LogisticRegressionModelRemoteStub.serializeObj);

class LinearRegressionModel:

    # initialize a LinearRegressionModel object with "model" attribute containing an actual LinearRegression object from the skLearn module
    def __init__(self,*args,**kwargs):
        self.model=LinearRegression(*args,**kwargs)

    # a function that returns the actual LinearRegression object which the called LinearRegressionModel object wraps around
    def get_model(self):
        return self.model

    def fit(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        self.model.fit(X,y,sample_weight)
        return self
    
    def get_params(self,deep=True):
        return self.model.get_params(deep)

    def predict(self,X):
        # if statement added to avoid converting TabularData twice when predict() is called inside score()
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict(X)

    def score(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)        
        return self.model.score(X,y,sample_weight)

    def set_params(self,**params):
        return self.model.set_params(**params)
    
    '''
    # for testing purposes
    def __getattribute__(self,item):
            logging.info("The function being called: "+str(item))
            if (item in ('fit','predict','model','get_model','score')):
                return super().__getattribute__(item)
    '''
    
    def __getattribute__(self,item):
        # if the called function/attribute does not require X,y tabularData conversion, get the attribute value by calling the function on the actual LinearRegression model in skLearn module
        
        # check if this object has the requested attribute
        try:
            return super().__getattribute__(item)
        except:
            pass;
        # otherwise fetch it from the actual linear regression object
        return getattr(self.model,item)        

        '''
        if (item not in ('model','get_model','extract_X','extract_y','fit','predict','score')):
            return getattr(self.model,item)
        # else, call the function/attribute defined in the local module
        else:
            return object.__getattribute__(self,item)
        '''

copyreg.pickle(LinearRegressionModel,LinearRegressionModelRemoteStub.serializeObj);	

class DecisionTreeModel:
    # initialize a DecisionTreeModel object with "model" attribute containing an actual DecisionTreeClassifier object from the skLearn module
    def __init__(self,*args,**kwargs):
        self.model = DecisionTreeClassifier(*args, **kwargs)

    def get_model(self):
        return self.model

    def apply(self,X,check_input=True):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.apply(X,check_input)

    def cost_complexity_pruning_path(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        return self.model.cost_complexity_pruning_path(X,y,sample_weight)        
    def decision_path(self,X,check_input=True):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.decision_path(X,check_input)
    
    def fit(self,X,y,sample_weight=None,check_input=True,X_idx_sorted=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        self.model.fit(X,y,sample_weight,check_input,X_idx_sorted)
        return self

    def predict(self,X,check_input=True):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict(X,check_input)

    def predict_log_proba(self,X):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict_log_proba(X)

    def predict_proba(self,X,check_input=True):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        return self.model.predict_proba(X,check_input)

    def score(self,X,y,sample_weight=None):
        if (isinstance(X,TabularData)):
            X=DataConversion.extract_X(X)
        if (isinstance(y,TabularData)):
            y=DataConversion.extract_y(y)
        return self.model.score(X,y,sample_weight)

    def __getattribute__(self,item):
        try:
            return super().__getattribute__(item)
        except:
            pass;
        return getattr(self.model,item)

copyreg.pickle(DecisionTreeModel,DecisionTreeModelRemoteStub.serializeObj);

class HelloWorld(metaclass=ABCMeta):
    def _helloWorld(self):
        logging.info("Hello World")
copyreg.pickle(HelloWorld,HelloWorldRemoteStub.serializeObj);

class DBC(metaclass=ABCMeta):
    _dataFrameClass_ = None;

    class ModelRepository:
        def __init__(self,dbc):
            self.dbc = dbc

        def __getattribute__(self,item):
            try:
                return object.__getattribute__(self,item)
            except:
                pass
            logging.info("_load('{}')".format(item))
            m=self.dbc._load('{}'.format(item))
            logging.info(type(m))
            self.__setattr__(item,m)
            return object.__getattribute__(self,item)

    class SQLTYPE(Enum):
        SELECT=1; CREATE=2; DROP=3; INSERT=4; DELETE=5;

    class AGGTYPE(Enum):
        SUM='SUM({})'; AVG='AVG({})';
        COUNT='COUNT({})'; COUNTD='COUNT(DISTINCT {})'; COUNTN='SUM(CASE WHEN {} IS NULL THEN 1 ELSE 0 END)';
        MAX='MAX({})'; MIN='MIN({})';

    _tableRepo_ = weakref.WeakValueDictionary();
    _plotURLRepo_ = {};

    def __init__(self, conMgr, jobName, dbName, serverIPAddr):
        self._conMgr = conMgr;
        self._jobName = jobName;
        self._conMgr.add(jobName, self);
        self._roMgrObj = ROMgr.getROMgr();
        self._schMgr = scheduler.getScheduleManager();
        #self._schMgr = aidasys.schMgr;
        self._dbName = dbName;
        self._serverIPAddr = serverIPAddr;
        self._workSpaceProxies_ = {};
        self._webDivIds = {};
        self._models = DBC.ModelRepository(self)

    #@abstractmethod
    #def _getDBTable(self, relName, dbName=None): pass;

    @property
    def dbName(self):
        return self._dbName;

    #Give a list of tables.
    @abstractmethod
    def _tables(self): pass;

    #This has to be a class method or the TableUDFs cannot get hold of this function.
    @classmethod
    def _getDBTable(cls, relName, dbName=None):
        return cls._tableRepo_[relName];

    @abstractmethod
    def _executeQry(self, sql, resultFormat, sqlType): pass;

    def _Page(self, func, *args, **kwargs):
        if(isinstance(func, str)):
            func = super().__getattribute__(func);

        appObj = GBackendApp.getGBackendAppObj();
        plotLayout =  func(weakref.proxy(self), appObj.app, *args, **kwargs);
        plotURL = GBackendApp.genURLPath(self._jobName);
        self._plotURLRepo_[plotURL] = plotLayout;
        appObj.addURL(plotURL,self);

        if(AConfig.PAGETUNNEL is None):
            return 'http://' + self._serverIPAddr + ':' + str(AConfig.DASHPORT) + plotURL;
        else:
            return 'https://' + AConfig.PAGETUNNEL + plotURL;

    def genDivId(self, id):
        #func = inspect.stack()[1][3];
        #divId =  self._jobName + '-' + func + '-' + id + '-' + hex(abs(hash((self._jobName, func, id, random.random()))));
        divId =  self._jobName + '-' + id + '-' + hex(abs(hash((self._jobName, id, random.random()))));
        self._webDivIds[id] = divId;
        return divId;

    def getDivId(self, id):
        return self._webDivIds[id];

    def _Plot(self, func, *args, **kwargs):
        """Function that is called from stub to execute a Dash graph plotting python function in this workspace"""
        #Execute the function with this workspace as the argument and return the results if any.
        if(isinstance(func, str)):
            func = super().__getattribute__(func);

        plotData =  func(self, *args, **kwargs);

        if(isinstance(plotData, dict)):
            #plotLayout =  GBackendApp.wrapGraph(func(self, *args, **kwargs));
            plotLayout =  GBackendApp.wrapGraph(plotData);
            plotURL = GBackendApp.genURLPath(self._jobName);
            self._plotURLRepo_[plotURL] = plotLayout;
            GBackendApp.getGBackendAppObj().addURL(plotURL,self);

            if(AConfig.PAGETUNNEL is None):
                return 'http://' + self._serverIPAddr + ':' + str(AConfig.DASHPORT) + plotURL;
            else:
                return 'https://' + AConfig.PAGETUNNEL + plotURL;
        else:
            imgData = BytesIO();
            plotData.savefig(imgData, format='png');
            return imgData;

    def getPlotLayout(self, plotURL):
        """Function that is invoked by the GBackendApp to obtain the layout of a plot"""
        return self._plotURLRepo_[plotURL];

    def _X(self, func, *args, **kwargs):
        """Function that is called from stub to execute a python function in this workspace"""
        #Execute the function with this workspace as the argument and return the results if any.
        if(isinstance(func, str)):
            func = super().__getattribute__(func);
        start_time = time.time();
        pred = func(TensorWrap_CPU(self), *args, **kwargs);
        end_time = time.time();
        rt = end_time - start_time;
        result = "time: "+ str(rt)
        return result;
        #return func(self, *args, **kwargs);

    def _append(self,iter_func,cond_func,test_func,name,*args,**kwargs):
        """Function that is called from stub to execute a python function in this workspace between cpu and gpu"""
        #Execute the function with this workspace as the argument and return the results if any.
        if(isinstance(iter_func, str)):
            iter_func = super().__getattribute__(iter_func);
        if(isinstance(cond_func, str)):
            cond_func = super().__getattribute__(cond_func);
        use_gpu = False
        time_limit = 5
        first_round = True
        self.stop = False
        start_t = time.time();

        condition = threading.Condition();
        with condition:
            logging.info(condition);
            self._schMgr.append_GPU(condition,name,self);
            condition.wait();

        st = time.time()
        logging.info(name+"start running at"+str(st))
        iter_func(TensorWrap_GPU(self),self.epoch_total,float('inf'),True, *args, **kwargs);
        test_func(TensorWrap_GPU(self),True, *args, **kwargs);
        en = time.time()
        self._schMgr.finish_GPU_all(self)
        end_t = time.time();

        logging.info(name+"end running at"+str(end_t))
        #return ed_tnd_t - st;
        return end_t - start_t;

    def _append_no_iter(self,iter_func,cond_func,test_func,name,*args,**kwargs):
        """Function that is called from stub to execute a python function in this workspace between cpu and gpu"""
        #Execute the function with this workspace as the argument and return the results if any.
        if(isinstance(iter_func, str)):
            iter_func = super().__getattribute__(iter_func);
        if(isinstance(cond_func, str)):
            cond_func = super().__getattribute__(cond_func);
        use_gpu = False
        time_limit = 5
        first_round = True
        self.stop = False
        start_t = time.time();

        condition = threading.Condition();
        with condition:
            logging.info(condition);
            self._schMgr.append_GPU(condition,name,self);
            condition.wait();

        st = time.time()
        logging.info(name+"start running at"+str(st))
        while(not(cond_func(TensorWrap_GPU(self),True, *args, **kwargs))):
            iter_func(TensorWrap_GPU(self),1000000,5,True, *args, **kwargs);
        test_func(TensorWrap_GPU(self),True, *args, **kwargs);
        en = time.time()
        self._schMgr.finish_GPU_all(self)
        end_t = time.time();

        logging.info(name+"end running at"+str(end_t))
        return end_t - start_t;

    def _Preprocess(self, func, dataset, *args, **kwargs):
        if(isinstance(func, str)):
            func = super().__getattribute__(func);
        self.dataset = dataset
        start_time = time.time();
        #self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        processed_data  = func(self, dataset, *args, **kwargs);

        # Record processed dataset in dw instance.

        end_time = time.time();
        rt = end_time - start_time;
        result = "time: "+ str(rt)
        #return result;
        return processed_data;

    def _KMeans(self, epochs, clusters):
        self.epoch_total = epochs
        self.K = clusters
        self.cl, self.c = 0,0
        self.epoch_done = 0
        def k_means(dw, time_limit, x, c=0, cl=0,  K=10, Niter=10, verbose=False, if_gpu=True):
            start = time.time()
            N, D = x.shape  # Number of samples, dimension of the ambient space
            num_finish = -1

            if isinstance(c, int) and c == 0:
                c = x[:K, :].clone()  # Simplistic initialization for the centroids
                logging.info(' x size'+str(x.size()))
                logging.info('c size'+str(c.size()))
                logging.info('K is'+str(K))
            elif c.size()[1] != 2:
                c = torch.transpose(c,0,1)
            if if_gpu:
                x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
                logging.info('size'+str(c.size()))
                c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids
            else:
                x_i = x.view(N, 1, D)  # (N, 1, D) samples
                c_j = c.view(1, K, D)  # (1, K, D) centroids

            # K-means loop:
            # - x  is the (N, D) point cloud,
            # - cl is the (N,) vector of class labels
            # - c  is the (K, D) cloud of cluster centroids
            for i in range(Niter):

                if(time.time() - start < time_limit):
                    # E step: assign points to the closest cluster -------------------------
                    D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
                    cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
                    #logging.info("cl"+str(cl))

                    # M step: update the centroids to the normalized cluster average: ------
                    # Compute the sum of points per cluster:
                    c.zero_()
                    c.scatter_add_(0, cl[:, None].repeat(1, D), x)

                    # Divide by the number of points per cluster:
                    Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K,1)
                    c /= Ncl  # in-place division to compute the average
                else:
                    num_finish = i;
                    break;
                if(dw.stop):
                    num_finish = i;
                    break;
            if(num_finish == -1):
                num_finish = Niter;

            return cl, c,num_finish

        def trainingLoop(dw,iter_num,time_limit,using_GPU):
            start = time.time()
            if(dw.reschedule):
                gpus = GPUtil.getGPUs()
                return [(time.time() - start),num_finish,gpus[1].load*100,psutil.cpu_percent()]
            cl = dw.cl
            c = dw.c
            print(c)
            K = dw.K
            x = dw.x
            Niter = iter_num
            cl, c,num_finish = k_means(dw,time_limit,x, c, cl, K, Niter=Niter, if_gpu = using_GPU)
            dw.cl = cl
            dw.c = c
            end = time.time()
            #dw.timeArray.append(end-start)
            dw.epoch_done += num_finish
            #dw.finishedEpochArray.append(dw.epoch_done)
            gpus = GPUtil.getGPUs()
            return [(time.time() - start),num_finish,gpus[1].load*100,psutil.cpu_percent()]

        def Condition(dw,use_GPU):
            if dw.epoch_done >= dw.epoch_total:
                return True
            else:
                return False

        def Testing(dw,using_GPU):
            pass
        return self._job(trainingLoop, Condition, Testing, "kmeans")

    def _NN(self, model, forward, criterion, optimizer, epochs, name, loss_limit=None, *args, **kwargs):
        self.epoch_done = 0
        self.loss_limit = loss_limit
        if(epochs != None):
            self.epoch_total = epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        #self.epochs = epochs
        self.stop = False
        self.loss = 100000000000000
        self.loss_arr = []
        model.forward = MethodType(forward, model)
        def iterate(dw,iter_num,time_limit,using_GPU):
            psutil.cpu_percent()
            model = dw.model
            #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
            optimizer = dw.optimizer
            x_train = dw.x_train
            y_train  = dw.y_train
            criterion = dw.criterion
            start = time.time()
            num_finish = 0;

            if(dw.reschedule):
                gpus = GPUtil.getGPUs()
                return [(time.time() - start),num_finish,gpus[1].load*100,psutil.cpu_percent()]
            for i in range (iter_num):
                if(time.time() - start < time_limit):
                    model = self.model
                    predicted = model(x_train)
                    loss = criterion(predicted, y_train)
                    self.loss = loss.item()
                    self.loss_arr.append(self.loss)
                    if self.loss_limit != None and self.loss <= self.loss_limit:
                        dw.stop = True
                    else:
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        self.model = model

                else:
                    num_finish = i;
                    break;
                if(dw.stop):
                    num_finish = i;
                    logging.info("stopped")
                    break;

            if(num_finish == 0):
                num_finish = iter_num;
            dw.model = model
            epoch_done = dw.epoch_done + num_finish
            dw.epoch_done = epoch_done
            gpus = GPUtil.getGPUs()
            return [(time.time() - start),num_finish,gpus[1].load*100,float(psutil.cpu_percent())]

        def condition(dw, using_GPU):
            dw.testFunc = 1
            logging.info("testFunc: " + str(dw.testFunc))
            if dw.loss_limit == None:
                if dw.epoch_done >= dw.epoch_total:
                    return True
                else:
                    return False

            else:
                if dw.loss <= dw.loss_limit:
                    return True
                else:
                    return False

        def test_model(dw, using_GPU):

           #if self.x_test == None or self.y_test == None:
               #return "No x_test or y_test provided"
            x_test = dw.x_test
            y_test = dw.y_test
            predicted = dw.model(x_test)
            loss = dw.criterion(predicted,y_test)
            return_mesg = ""
            return return_mesg

        return self._job(iterate, condition, test_model, name)


    def _job(self,iter_func,cond_func,test_func,name,*args,**kwargs):
        """Function that is called from stub to execute a python function in this workspace between cpu and gpu"""
        #Execute the function with this workspace as the argument and return the results if any.
        logging.info(name+" arrive at "+str(time.time()))
        if(isinstance(iter_func, str)):
            iter_func = super().__getattribute__(iter_func);
        if(isinstance(cond_func, str)):
            cond_func = super().__getattribute__(cond_func);
        use_gpu = False
        first_round = True
        condition = threading.Condition();
        self.stop = False
        self.reschedule = False
        LARGE_NUM =  10000000000 
        time_stamp = time.time()
        epoch_total = LARGE_NUM # a random large int
        try:
            epoch_total =self.epoch_total
        except:
            logging.info("no explicit epoch num")
        start_t = time.time();

        logging.info("enter job")
        self._schMgr.coming_test_gpu();
        gpus = GPUtil.getGPUs()
        wait_gpu_start = time.time()
        while(gpus[1].load * 100 > 20 or psutil.cpu_percent(interval=0.5)>20):
            time.sleep(1)
            self._schMgr.coming_test_gpu();
            gpus = GPUtil.getGPUs()
        wait_gpu_end = time.time()
        #logging.info("time for waiting gpu test:" + str( wait_gpu_end- wait_gpu_start))
        #time_limit = 1
        time_limit = self._schMgr.get_GPU_running_job_length()/10
        if(time_limit < 1 ):
            time_limit = 1
        start_gpu = time.time()
        logging.info("test start at:"+ str(start_gpu))
        GPU_test=iter_func(TensorWrap_GPU(self),epoch_total,time_limit,True, *args, **kwargs);
        logging.info("gpu util:"+str(GPU_test))
        en_gpu = time.time()
        #logging.info("time for gpu test:" + str(en_gpu-start_gpu));
        percent_finished = float(GPU_test[1]/epoch_total);
        CPU_test = [-1,-1,-1,-1] 
        #if the job is short enough to be done in the time_limit on GPU
        if(percent_finished == 1 or cond_func(TensorWrap_GPU(self),True, *args, **kwargs)):
            result = test_func(TensorWrap_GPU(self),True, *args, **kwargs);
            self._schMgr.finish_test_gpu();
            end_t = time.time();
            return end_t-start_t;
        #if the job is relatively short, meaning time_limit has finished large part of it, say 40%,
        #put it directly to GPU queue with estimate time
        elif(epoch_total != LARGE_NUM and self.epoch_total > 0 and percent_finished > 0.3):
            self._schMgr.finish_test_gpu();
            estimate_time = GPU_test[0] *(self.epoch_total - self.epoch_done)/GPU_test[1];
            #logging.info(name+" wait for schedule(most part) "+str(time.time()))
            with condition:
                self._schMgr.insert_GPU(condition,name,self,estimate_time,GPU_test[2],GPU_test[3],time_stamp);
                condition.wait();
        else:
            logging.info("test on cpu")
            #need to do the CPU test
            self._schMgr.coming_test_cpu();
            psutil.cpu_percent()
            time.sleep(0.2)
            per = psutil.cpu_percent()
            #when current job is not stopped, keep waiting until no one is using cpu
            while(per > 20):
                #logging.info("cpu:" + str(per))
                #logging.info("wait for starting test on cpu")
                time.sleep(1)
                self._schMgr.coming_test_cpu();
                per = psutil.cpu_percent()
            st2 = time.time()
            time_limit = 1
            #time_limit = self._schMgr.get_CPU_running_job_length()/10
            #if(time_limit < 1):
            #    time_limit = 1
            CPU_test=iter_func(TensorWrap_CPU(self),epoch_total,time_limit,False,*args, **kwargs);
            logging.info("cpu util:"+str(CPU_test))
            en2 = time.time()
            #logging.info("time for cpu test:" + str(en2-st2));
            #logging.info("time till cpu test:" + str(en2-start_t));
            if(epoch_total != LARGE_NUM):
                #known total epoches
                estimate_time_GPU = GPU_test[0] *(self.epoch_total - self.epoch_done)/GPU_test[1];
                estimate_time_CPU = CPU_test[0] * (self.epoch_total - self.epoch_done)/CPU_test[1];
                exec_on_CPU_time = self._schMgr.get_CPU_waiting(estimate_time_CPU);
                exec_on_GPU_time = self._schMgr.get_GPU_waiting(estimate_time_GPU);
                logging.info("cpu"+ str(CPU_test[0] *self.epoch_total/CPU_test[1]))
                logging.info("gpu"+ str(GPU_test[0] *self.epoch_total/GPU_test[1]))
                if(float(exec_on_GPU_time/exec_on_CPU_time) > 1):
                    logging.info("need to go cpu queue")
                    self._schMgr.insert_CPU(condition,name,self,estimate_time_CPU,CPU_test[2],CPU_test[3],time_stamp);
                else:
                    logging.info("need to go gpu queue")
                    self._schMgr.insert_GPU(condition,name,self,estimate_time_GPU,GPU_test[2],GPU_test[3],time_stamp);
            else:
                #unknown
                if(float(CPU_test[1]/CPU_test[0]) > float(GPU_test[1]/GPU_test[0])):
                    self._schMgr.insert_CPU_long(condition,name,self,CPU_test[2],CPU_test[3]);
                else:
                    self._schMgr.insert_GPU_long(condition,name,self,GPU_test[2],GPU_test[3]);
            self._schMgr.finish_test_gpu();
            self._schMgr.finish_test_cpu();
            logging.info(name+" wait for schedule(after cpu test) "+str(time.time()))
            with condition:
                condition.wait();


        #after being waken up
        def run(func,on_GPU,*args):
            if(on_GPU):
                logging.info(name + "run on gpu")
                result = func(TensorWrap_GPU(self),*args,True, **kwargs);
                return result;
            else:
                logging.info(name + "run on cpu")
                result= func(TensorWrap_CPU(self),*args, False, **kwargs);
                return result;

        def finish(on_GPU,finish_all):
            if(finish_all):
                if(on_GPU):
                    self._schMgr.finish_GPU_all(self)
                else:
                    self._schMgr.finish_CPU_all(self)
            else:
                if(on_GPU):
                    self._schMgr.finish_GPU_pause(self,name)
                else:
                    self._schMgr.finish_CPU_pause(self,name)
        
        on_GPU = self._schMgr.in_GPU(self);
        st3 = time.time()
        #result:[time,epoch,cpu util]
        if(epoch_total != LARGE_NUM):
            logging.info(name+"run at "+str(time.time()))
            result = run(iter_func,on_GPU,self.epoch_total - self.epoch_done,float('inf'))  
        else:
            if(on_GPU):
                time_limit = self._schMgr.get_GPU_running_job_length()
            else:
                time_limit = self._schMgr.get_CPU_running_job_length()
            if(time_limit < 0):
                time_limit = 5
            logging.info(name+"run for"+str(time_limit)+" at "+str(time.time()))
            result = run(iter_func,on_GPU,epoch_total,time_limit)
        while(not run(cond_func,on_GPU)):
            finish(on_GPU,False);
            logging.info(name+"paused at "+str(time.time()))
            st4 = time.time()
            #need to check the testing currently on GPU has finished and the 
            #corresponding dw has beed added to the queue, then it has the right place to 
            #put itself in the queue
            while(self.stop):
                time.sleep(0.3)

            # recalculate the time
            if(CPU_test[0] == -1):
                estimate_time = GPU_test[0] *(self.epoch_total - self.epoch_done)/GPU_test[1];
                self._schMgr.insert_GPU(condition,name,self,estimate_time,GPU_test[2],GPU_test[3],time_stamp);

            elif(epoch_total != LARGE_NUM):
                estimate_time_CPU = CPU_test[0] * (self.epoch_total - self.epoch_done)/CPU_test[1];
                estimate_time_GPU = GPU_test[0] *(self.epoch_total - self.epoch_done)/GPU_test[1];
                exec_on_CPU_time = self._schMgr.get_CPU_waiting(estimate_time_CPU);
                exec_on_GPU_time = self._schMgr.get_GPU_waiting(estimate_time_GPU);
                if(float(exec_on_GPU_time/exec_on_CPU_time) > 1):
                    self._schMgr.insert_CPU(condition,name,self,estimate_time_CPU,CPU_test[2],CPU_test[3],time_stamp);
                else:
                    self._schMgr.insert_GPU(condition,name,self,estimate_time_GPU,GPU_test[2],GPU_test[3],time_stamp);
            else:
                if(not on_GPU):
                    self._schMgr.insert_CPU_long(condition,name,self,CPU_test[2],CPU_test[3]);
                else:
                    self._schMgr.insert_GPU_long(condition,name,self,GPU_test[2],GPU_test[3]);

            if(on_GPU):
                self._schMgr.put_back_GPU_to_queue(self)
            else:
                self._schMgr.put_back_CPU_to_queue(self)
            with condition:
                condition.wait();


            en4 = time.time()
            logging.info("time for waiting:" + str(en4-st4));
            logging.info(name+"run for"+str(time_limit)+" at "+str(time.time()))
            if(epoch_total != LARGE_NUM):
                result = run(iter_func,on_GPU,self.epoch_total - self.epoch_done,float('inf'))
            else:
                if(on_GPU):
                    time_limit = self._schMgr.get_GPU_running_job_length()
                else:
                    time_limit = self._schMgr.get_CPU_running_job_length()
                if(time_limit < 0):
                    time_limit = 5
                logging.info("run for"+str(time_limit))
                result = run(iter_func,on_GPU,epoch_total,time_limit)
                #iter_func(TensorWrap_GPU(self),self.epoch_total - self.epoch_done,float('inf'),True, *args, **kwargs);
        en3 = time.time()
        #logging.info("time for running remaining:" + str(en3-st3));
        result = run(test_func,on_GPU) 
        finish(on_GPU,True);
        end_t = time.time();
        logging.info(name+"terminate at "+str(time.time()))

        return end_t - start_t;

    def _XP(self, func, *args, **kwargs):
        """Function that is called from stub to execute a python function in this workspace"""
        #Execute the function with this workspace as the argument and return the results if any.
        #Wrap the DBC object to make sure that the DBC object returns only NumPy matrix representations of the TabularData objects.
        #TODO: Go over the args and kwargs and replace TabularData objects with NumPy matrix representations.
        if(isinstance(func, str)):
            func = super().__getattribute__(func);
        return func(DBCWrap(self), *args, **kwargs);

    def _GPU(self, func, *args, **kwargs):
        """Function that is called from stub to execute a python function in this workspace"""
        #Execute the function with this workspace as the argument and return the results if any.
        #Wrap the DBC object to make sure that the DBC object returns only CuPy matrix representations of the TabularData objects.
        if(isinstance(func, str)):
            func = super().__getattribute__(func);
        #func(GPUWrap(self), *args, **kwargs)
        #torch.cuda.cudart().cudaProfilerStart();
        #func(GPUWrap(self), *args, **kwargs);
        #torch.cuda.cudart().cudaProfilerStop()
        return func(GPUWrap(self), *args, **kwargs);

    def _helloWorld(self):
        hw=HelloWorld()
        return hw        

    def _linearRegression(self,*args,**kwargs):
        model=LinearRegressionModel(*args,**kwargs)
        return model

    def _logisticRegression(self,*args,**kwargs):
        model=LogisticRegressionModel(*args,**kwargs)
        return model

    def _decisionTree(self,*args,**kwargs):
        model=DecisionTreeModel(*args,**kwargs)
        return model

    def _save(self,model_name,model,update=False):
        
        # Code using MERGE 
        '''
        m = model.get_model()
        model_type=type(m)
        pickled_m = pickle.dumps(m)
        pickled_m = str(pickled_m)
        pickled_m = pickled_m.replace("'","''")
        pickled_m = pickled_m.replace("\\","\\\\")

        if (update==True):
            self._executeQry("MERGE INTO _sys_models_ AS to_update USING _sys_models_ AS models_update ON to_update.model_name = models._update.model_name WHEN MATCHED THEN DELETE;",sqlType=DBC.SQLTYPE.MERGE)
            self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name,pickled_m,model_type),sqlType=DBC.SQLTYPE.INSERT)
        # update==False
        else:
            try:
                self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name,pickled_m,model_type),sqlType=DBC.SQLTYPE.INSERT)
            except:
                raise Exception("There already exists a model in the database with the same model_name. Please set \'update\' to True to overwrite" )        

        '''   
        
        duplicate_exist = False
        logging.info("before exeucing query")
        # check if there is another model already saved with <model_name>
        temp = self._executeQry("SELECT COUNT(model) as count FROM _sys_models_ WHERE model_name='{}';".format(model_name))
        # if the above SELECT COUNT query returns integer not equal to 0
        if temp[0]['count'][0]!=0:
            duplicate_exist = True
        logging.info("after executing query once , and check duplicate")
    
        # throw an error if update=False and there is another model already saved with <model_name>
        if (update==False and duplicate_exist==True):
            raise Exception("There already exists a model in the database with the same model_name. Please set \'update\' to True to overwrite" ) 
        # delete the model saved with <model_name> if update=True
        elif (update==True and duplicate_exist==True):
            self._executeQry("DELETE FROM _sys_models_ WHERE model_name='{}';".format(model_name),sqlType=DBC.SQLTYPE.DELETE)
        else:
            pass

        m = model.get_model()
        model_type = type(m)
        model_type = str(model_type)
        model_type = model_type.replace("'","''")

        pickled_m = pickle.dumps(m)
        pickled_m = str(pickled_m)
        pickled_m = pickled_m.replace("'","''")

        if AConfig.DATABASEADAPTER == "aidaMonetDB.dbAdapter.DBCMonetDB":
            pickled_m = pickled_m.replace("\\","\\\\")
        elif AConfig.DATABASEADAPTER == "aidaPostgreSQL.dbAdapter.DBCPostgreSQL":
            pass

        self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name,pickled_m,model_type),sqlType=DBC.SQLTYPE.INSERT)

    def _saveTorchModel(self, model_name, model, update=False):

        # Code using MERGE
        '''
        m = model.get_model()
        model_type=type(m)
        pickled_m = pickle.dumps(m)
        pickled_m = str(pickled_m)
        pickled_m = pickled_m.replace("'","''")
        pickled_m = pickled_m.replace("\\","\\\\")

        if (update==True):
            self._executeQry("MERGE INTO _sys_models_ AS to_update USING _sys_models_ AS models_update ON to_update.model_name = models._update.model_name WHEN MATCHED THEN DELETE;",sqlType=DBC.SQLTYPE.MERGE)
            self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name,pickled_m,model_type),sqlType=DBC.SQLTYPE.INSERT)
        # update==False
        else:
            try:
                self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name,pickled_m,model_type),sqlType=DBC.SQLTYPE.INSERT)
            except:
                raise Exception("There already exists a model in the database with the same model_name. Please set \'update\' to True to overwrite" )

        '''

        duplicate_exist = False
        logging.info("before exeucing query")
        # check if there is another model already saved with <model_name>
        temp = self._executeQry(
            "SELECT COUNT(model) as count FROM _sys_models_ WHERE model_name='{}';".format(model_name))
        # if the above SELECT COUNT query returns integer not equal to 0
        if temp[0]['count'][0] != 0:
            duplicate_exist = True
        logging.info("after executing query once , and check duplicate")

        # throw an error if update=False and there is another model already saved with <model_name>
        if (update == False and duplicate_exist == True):
            raise Exception(
                "There already exists a model in the database with the same model_name. Please set \'update\' to True to overwrite")
            # delete the model saved with <model_name> if update=True
        elif (update == True and duplicate_exist == True):
            self._executeQry("DELETE FROM _sys_models_ WHERE model_name='{}';".format(model_name),sqlType=DBC.SQLTYPE.DELETE)
        else:
            pass

        model_type = type(model)
        model_type = str(model_type)
        model_type = model_type.replace("'","''")

        pickled_m = pickle.dumps(model)
        pickled_m = str(pickled_m)
        pickled_m = pickled_m.replace("'","''")

        if AConfig.DATABASEADAPTER == "aidaMonetDB.dbAdapter.DBCMonetDB":
            pickled_m = pickled_m.replace("\\", "\\\\")
        elif AConfig.DATABASEADAPTER == "aidaPostgreSQL.dbAdapter.DBCPostgreSQL":
            pass

        self._executeQry("INSERT INTO _sys_models_ VALUES('{}','{}','{}');".format(model_name, pickled_m, model_type),sqlType=DBC.SQLTYPE.INSERT)

    def _loadTorchModel(self,model_name):

        unpickled_m = self._executeQry("SELECT model FROM _sys_models_ WHERE model_name = '{}';".format(model_name))
        try:
            unpickled_m = unpickled_m[0]['model'][0]
        except:
            raise Exception("no model with such name found.")

        model=pickle.loads(ast.literal_eval(unpickled_m))
        logging.info(type(model))
        # default as linear regression model
        return model

    # testing sql
    def _sql(self,sql):
        logging.info(self._executeQry(sql))

    def _L(self, func, *args, **kwargs):
        return DBC._dataFrameClass_._loadExtData_(func, self, *args, **kwargs);

    def _ones(self, shape, cols=None):
        return DBC._dataFrameClass_.ones(shape, cols, self);

    def _rand(self, shape, cols=None):
        return DBC._dataFrameClass_.rand(shape, cols, self);

    def _randn(self, shape, cols=None):
        return DBC._dataFrameClass_.randn(shape, cols, self);

    @abstractmethod
    def _toTable(self, tblrData, tableName=None): pass;

    @abstractmethod
    def _saveTblrData(self, tblrData, tableName, dbName=None, drop=False): pass;

    @abstractmethod
    def _dropTable(self, tableName, dbName=None): pass;

    @abstractmethod
    def _dropTblUDF(self, tblrData, tableName=None): pass;

    @abstractmethod
    def _describe(self, tblrData): pass;

    @abstractmethod
    def _agg(self, agfn, tblrData, collist=None, valueOnly=True): pass;

    def _close(self):
        self._conMgr.remove(self._jobName);

    def _registerProxy_(self, attrname, proxyid):
        """"Invoked by the remote stub of DBC to register the proxyid of a stub it has pointing to an object in this workspace """
        proxyObj = self._roMgrObj.getProxy(proxyid);
        try:
            self._workSpaceProxies_[attrname].add(proxyObj);
        except KeyError:
            self._workSpaceProxies_[attrname] = weakref.WeakSet([proxyObj,]);

    def __getattribute__(self, item):
        #Check if the object has this attribute.
        try:
            return super().__getattribute__(item);
        except:
            pass;
        #Maybe its a database object.
        #TODO: Need to ensure that databaseworkspace.var = tabularDataObject does not work if var is a valid tablename in db.
        try:
            #Check if we already have this table loaded up in the DB workspace.
            return super(self.__class__, self)._getDBTable(item);
        except KeyError:
            #We do not have it, so let us see if the adapter implementation class can load it from the database.
            tbl = self._getDBTable(item);
            #Store the table handle for any future use.
            self._tableRepo_[item] = tbl;
            return tbl;

    def _setattr_(self, key, value, returnAttr=False):
        """Invoked by the remote DBC to set an attribute in DBC for use in workspace"""
        #logging.debug("DBC workspace : to set {} to value of type {}".format(key, type(value)));
        try: #First see if we already have an attribute.
            cur =  self.__getattribute__(key);
        except:
            cur = None;

        if(cur is not None):
            if(isinstance(cur, DBObject)):
                logging.error("Error: there is already an object {} in the database.".format(key));
                raise TypeError("Error: there is already an object {} in the database.".format(key));
            #If there is already an attribute, make sure that its stub serialization procedures match the new one.
            #Or current one has no special stub serialization. otherwise raise an exception.
            curtype = copyreg.dispatch_table.get(type(cur));
            valuetype = copyreg.dispatch_table.get(type(value));
            #logging.debug("_setattr_ called for an existing attribute {} curtype {} valuetype {}".format(key, curtype, valuetype));
            if(curtype is not None and curtype != valuetype):
                logging.error("Error: unable to set {} remote stub for new type {} does not match that of the current type {}".format(key, type(key), type(cur)));
                raise TypeError("Error: unable to set {} remote stub for new type {} does not match that of the current type {}".format(key, type(key), type(cur)));
            #If current attributes's type is one with a remote stub assigned,
            # replace the reference of current proxies with the new obj.
            if(curtype is not None):
                proxies = self._workSpaceProxies_.get(key);
                if(proxies is not None):
                    #logging.debug("Replacing object in place for attribute {}".format(key));
                    self._roMgrObj.replace(proxies, value);

        #Do the actual setting of the attribute.
        super().__setattr__(key, value);
        if(returnAttr):
            return value;

    def __setattr__(self, key, value):
        if(key.startswith('_')):
            return super().__setattr__(key, value);
        self._setattr_(key, value, False);

    def __delattr__(self, item):
        logging.debug("DBC: __delattr__ : have to remove attribute : {}.".format(item));
        if(item.startswith('_')):
            return super().__delattr__(item);
        try:
            del self._tableRepo_[item];
            logging.debug("DBC: __delattr__ : removed {} from tableRepo.".format(item));
        except KeyError:
            pass;
        try:
            super().__delattr__(item);
        except:
            pass;
        self._dropTable(item);

copyreg.pickle(DBC.ModelRepository,DBCRemoteStub.ModelRepositoryRemoteStub.serializeObj);
copyreg.pickle(DBC, DBCRemoteStub.serializeObj);

###-###class DBCRemoteStub(aidacommon.rop.RObjStub):
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _getDBTable(self, relName, dbName=None):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _executeQry(self, sql, resultFormat='column'):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _x(self, func, *args, **kwargs):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _ones(self, shape, cols=None):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _rand(self, shape, cols=None):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _randn(self, shape, cols=None):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _toTable(self, tblrData, tableName=None):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _save(self, tblrData, tableName, dbName=None, drop=False):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _close(self):
###-###        pass;
###-###
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _registerProxy_(self, attrname, proxyid):
###-###        pass;
###-###
###-###    #TODO: write corresponding code in the DBC class
###-###    @aidacommon.rop.RObjStub.RemoteMethod()
###-###    def _setattr_(self, key, value, returnAttr=False):
###-###        pass;
###-###
###-###    def __getattribute__(self, item):
###-###        try:
###-###            #Check if we have the attribute locally.
###-###            return object.__getattribute__(self, item);
###-###        except:
###-###            pass;
###-###
###-###        #Find the object remotely ...
###-###        result = super().__getattribute__(item);
###-###
###-###        #If this a stub object, we are going to set it locally and also listen for updates on it.
###-###        if(isinstance(result, aidacommon.rop.RObjStub)):
###-###            self._registerProxy_(item, result.proxyid);
###-###            super().__setattr__(item, result);
###-###        #return the attribute.
###-###        return result
###-###
###-###    def __setattr__(self, key, value):
###-###        if(key.startswith('_')):
###-###            super().__setattr__(key, value);
###-###        else:
###-###            curval = None;
###-###            try:
###-###                #Find if there is an existing object for this key locally/remotely.
###-###                curval = self.__getattribute__(key);
###-###            except:
###-###                pass;
###-###
###-###            if(curval):
###-###                #If there is currently an object which is a remote object stub,
###-###                if(isinstance(curval, aidacommon.rop.RObjStub)):
###-###                    # but the new one is not, we cannot allow this.
###-###                    if(not isinstance(value, aidacommon.rop.RObjStub)):
###-###                        raise AttributeError("Error: cannot replace a remote stub with a regular object")
###-###                    #If we are replacing one remote obj stub with another, they need to have compatible stubs.
###-###                    if(not (isinstance(value, curval.__class__))):
###-###                        raise AttributeError("Error: the remote stubs are not compatible {} {}".format(value.__class__, curval.__class__))
###-###                #TODO if curval and value are the same, do nothing.
###-###
###-###            #Ask the remote object to set this attribute. it can also return a stub if this a stub and we do not have
###-###            #currently a stub pointing to it.
###-###            #TODO ... do we need to get stub back from do this ? why cannot we just use the existing stub ?
###-###            robj = self._setattr_(key, value, not(curval) and isinstance(value, aidacommon.rop.RObjStub) );
###-###            if(robj): #The remote object returned a stub.
###-###                self._registerProxy_(key, robj.proxyid);
###-###                super().__setattr__(key, robj);
###-###
###-###  /home/build/AIDA/aidacommon/dbA  #TODO: trap __del__ and call _close ?
###-###
###-###
###-###copyreg.pickle(DBCRemoteStub, DBCRemoteStub.serializeObj);

# Class to trap all calls to DBC from remote execution function if it only needs the NumPy objects.
# This class will trap any TabularData objects and pass out only their NumPy representation.
class DBCWrap:
    def __init__(self, dbcObj):
        self.__dbcObj__ = dbcObj; #This is the DBC workspace object we are wrapping
        self.__tDataColumns__ = {}; #Keep track of the column metadata of all TabularData variable names that we have seen so far.

    def __getattribute__(self, item):
        #Trap the calls to ALL my object variables here itself.
        if (item in ('__dbcObj__', '__tDataColumns__')):
            return super().__getattribute__(item);

        #Every other variable must come from the DBC object that we are wrapping.

        val = getattr(super().__getattribute__('__dbcObj__'), item);
        if(isinstance(val, TabularData)): #If the value returned from the DBC object is of type TabularData
            tDataCols = super().__getattribute__('__tDataColumns__');
            tDataCols[item] = val.columns; #We will keep track of that variable/objects metadata
            #Instead of returning the TabularData object, we will return only the NumPy matrix representation.
            #But since tabular data objects internally stored matrices in transposed format, we will have to transpose it
            # Back to regular format first.
            val = val.matrix.T;
            if(not val.flags['C_CONTIGUOUS']): #If the matrix is not C_CONTIGUOUS, make a copy in C_CONTGUOUS form.
                val = np.copy(val, order='C');
            if(len(val.shape) == 1):
                val = val.reshape(len(val), 1, order='C');
            #logging.debug("DBCWrap, getting : item {}, shape {}".format(item, val.shape));
        return val;

    def __setattr__(self, key, value):
        #Trap the calls to ALL my object variables here itself.
        if (key in ('__dbcObj__', '__tDataColumns__')):
            return super().__setattr__(key, value);

        #logging.debug("DBCWrap, setting called : item {}, value type {}".format(key, type(value)));
        #Every other variable is set inside the DBC object that we are wrapping.
        try:
            #logging.debug("DBCWrap: setattr : current known tabular data objects : {}".format(self.__tDataColumns__.keys()));

            #Convert numpy matrix back to TabularData object
            #Transpose the matrix to fit the internal form of TabularData objects.
            value = value.T;
            if(not value.flags['C_CONTIGUOUS']): #If the matrix is not C_CONTIGUOUS, make a copy in C_CONTGUOUS form.
                value = np.copy(value, order='C');
            #Build a new TabularData object using virtual transformation.
            #logging.debug("DBCWrap, setting : item {}, shape {}".format(key, value.shape));
            if key in self.__tDataColumns__:
                tDataCols = self.__tDataColumns__[key];
            #If we got to this line, then it means "key" was a TabularData object.
            # So we need to build a new TabularData object using the original column metadata.
                valueDF = DBC._dataFrameClass_._virtualData_(lambda:value, cols=tuple(tDataCols.keys()), colmeta=tDataCols, dbc=self.__dbcObj__);
            else:
            #If we go to this line, then it means "key" is a new variable.
            # So we need to build a new TabularData from scratch
                valueDF = DBC._dataFrameClass_._virtualData_(lambda:value, dbc=self.__dbcObj__);
            setattr(self.__dbcObj__, key, valueDF);
            return;
        except :
            logging.exception("DBCWrap : Exception ");
            pass;
        setattr(self.__dbcObj__, key, value);


# This class is very similar to DBCWrap class above except that here we are working on the CuPy objects instead of NumPy to accelarate the training process.
# To simplify, only the parts that are different from above have comments beside.
class GPUWrap:
    def __init__(self, dbcObj):
        self.__dbcObj__ = dbcObj; 
        self.__tDataColumns__ = {};

    def __getattribute__(self, item):
        if (item in ('__dbcObj__', '__tDataColumns__')):
            return super().__getattribute__(item);

        val = getattr(super().__getattribute__('__dbcObj__'), item);

        if(isinstance(val, TabularData)): 
            tDataCols = super().__getattribute__('__tDataColumns__');
            tDataCols[item] = val.columns;
            val = val.matrix.T;
            # val now is of type numpy.ndarray
            value_gpu = cp.asarray(val, order='C'); #convert to CuPy ndarray
            # val now is of type cupy.core.core.ndarray
            if(len(value_gpu.shape) == 1):
                value_gpu = value_gpu.reshape(len(value_gpu), 1, order='C');
            val = None;
            return value_gpu;

        return val;


    def __setattr__(self, key, value):
        if (key in ('__dbcObj__', '__tDataColumns__')):
            return super().__setattr__(key, value);

        try:
            value = value.T;
            # value now is of type cuoy.core.core.ndarray
            value_cpu = cp.asnumpy(value, order='C');

            if key in self.__tDataColumns__:
                tDataCols = self.__tDataColumns__[key];
            #If we got to this line, then it means "key" was a TabularData object.
            # So we need to build a new TabularData object using the original column metadata.
                valueDF = DBC._dataFrameClass_._virtualData_(lambda:value_cpu, cols=tuple(tDataCols.keys()), colmeta=tDataCols, dbc=self.__dbcObj__);
            else:
            #If we go to this line, then it means "key" is a new variable.
            # So we need to build a new TabularData from scratch
                valueDF = DBC._dataFrameClass_._virtualData_(lambda:value_cpu, dbc=self.__dbcObj__);
            setattr(self.__dbcObj__, key, valueDF);

            #remove cupy object and release gpu memory
            value = None;
            mempool = cp.get_default_memory_pool();
            pinned_mempool = cp.get_default_pinned_memory_pool();

            mempool.free_all_blocks();
            pinned_mempool.free_all_blocks();

            return;
        except :
            logging.exception("GPUWrap : Exception ");
            pass;
        setattr(self.__dbcObj__, key, value);
# This class is very similar to DBCWrap class above except that here we are working tensor->cupy objects instead of NumPy to accelarate the training process.
# To simplify, only the parts that are different from above have comments beside.
class TensorWrap_GPU:
    def __init__(self, dbcObj):
        self.__dbcObj__ = dbcObj; #This is the DBC workspace object we are wrapping
        self.__tDataColumns__ = {}; #Keep track of the column metadata of all TabularData variable names that we have seen so far.

    def __getattribute__(self, item):
        #Trap the calls to ALL my object variables here itself.
        if (item in ('__dbcObj__', '__tDataColumns__')):
            return super().__getattribute__(item);

        #Every other variable must come from the DBC object that we are wrapping.

        val = getattr(super().__getattribute__('__dbcObj__'), item);
        if(isinstance(val, TabularData)): #If the value returned from the DBC object is of type TabularData
            tDataCols = super().__getattribute__('__tDataColumns__');
            tDataCols[item] = val.columns; #We will keep track of that variable/objects metadata
            #Instead of returning the TabularData object, we will return only the NumPy matrix representation.
            #But since tabular data objects internally stored matrices in transposed format, we will have to transpose it
            # Back to regular format first.
            val = val.matrix.T;
            if(not val.flags['C_CONTIGUOUS']): #If the matrix is not C_CONTIGUOUS, make a copy in C_CONTGUOUS form.
                val = np.copy(val, order='C');
            if(len(val.shape) == 1):
                val = val.reshape(len(val), 1, order='C');
            #logging.debug("DBCWrap, getting : item {}, shape {}".format(item, val.shape));

        #directly create tensor on GPU    
        #if type(val) is np.ndarray:
        #    val = torch.tensor(val,device=torch.device("cuda:0"));
        #elif hasattr(val, 'to'):
        #    val = val.to(torch.device("cuda:0"));
        
        # create tensor on CPU then to GPU
        if type(val) is np.ndarray: 
            val = torch.tensor(val);
        if hasattr(val, 'to'):
            val = val.to(torch.device("cuda:0"));
        if isinstance(val,torch.optim.Optimizer):
            logging.info("is a optim")
            for param in val.state.values():
            # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(torch.device("cuda:0"))
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(torch.device("cuda:0"))
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(torch.device("cuda:0"))
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(torch.device("cuda:0"))
        return val;
    def __setattr__(self, key, value):
        #Trap the calls to ALL my object variables here itself.
        if (key in ('__dbcObj__', '__tDataColumns__')):
            return super().__setattr__(key, value);

        #logging.debug("DBCWrap, setting called : item {}, value type {}".format(key, type(value)));
        #Every other variable is set inside the DBC object that we are wrapping.
        try:
            #logging.debug("DBCWrap: setattr : current known tabular data objects : {}".format(self.__tDataColumns__.keys()));

            #Convert numpy matrix back to TabularData object
            #Transpose the matrix to fit the internal form of TabularData objects.
            if(isinstance(value,np.ndarray)):
                value = value.T;
            if(not value.flags['C_CONTIGUOUS']): #If the matrix is not C_CONTIGUOUS, make a copy in C_CONTGUOUS form.
                value = np.copy(value, order='C');
            #Build a new TabularData object using virtual transformation.
            #logging.debug("DBCWrap, setting : item {}, shape {}".format(key, value.shape));
            if key in self.__tDataColumns__:
                tDataCols = self.__tDataColumns__[key];
            #If we got to this line, then it means "key" was a TabularData object.
            # So we need to build a new TabularData object using the original column metadata.
                valueDF = DBC._dataFrameClass_._virtualData_(lambda:value, cols=tuple(tDataCols.keys()), colmeta=tDataCols, dbc=self.__dbcObj__);
            else:
            #If we go to this line, then it means "key" is a new variable.
            # So we need to build a new TabularData from scratch
                valueDF = DBC._dataFrameClass_._virtualData_(lambda:value, dbc=self.__dbcObj__);
            setattr(self.__dbcObj__, key, valueDF);
            return;
        except :
            logging.exception("DBCWrap : Exception ");
            pass;
        setattr(self.__dbcObj__, key, value);
# This class is very similar to DBCWrap class above except that here we are working numpy -> tensor objects instead of NumPy to accelarate the training process.
# To simplify, only the parts that are different from above have comments beside.
class TensorWrap_CPU:
    def __init__(self, dbcObj):
        self.__dbcObj__ = dbcObj; #This is the DBC workspace object we are wrapping
        self.__tDataColumns__ = {}; #Keep track of the column metadata of all TabularData variable names that we have seen so far.

    def __getattribute__(self, item):
        #Trap the calls to ALL my object variables here itself.
        if (item in ('__dbcObj__', '__tDataColumns__')):
            return super().__getattribute__(item);

        #Every other variable must come from the DBC object that we are wrapping.

        val = getattr(super().__getattribute__('__dbcObj__'), item);
        if(isinstance(val, TabularData)): #If the value returned from the DBC object is of type TabularData
            tDataCols = super().__getattribute__('__tDataColumns__');
            tDataCols[item] = val.columns; #We will keep track of that variable/objects metadata
            #Instead of returning the TabularData object, we will return only the NumPy matrix representation.
            #But since tabular data objects internally stored matrices in transposed format, we will have to transpose it
            # Back to regular format first.
            val = val.matrix.T;
            if(not val.flags['C_CONTIGUOUS']): #If the matrix is not C_CONTIGUOUS, make a copy in C_CONTGUOUS form.
                val = np.copy(val, order='C');
            if(len(val.shape) == 1):
                val = val.reshape(len(val), 1, order='C');
            #logging.debug("DBCWrap, getting : item {}, shape {}".format(item, val.shape));
        if type(val) is np.ndarray:
            val = torch.tensor(val);
        if hasattr(val, 'to'):
            val = val.to(torch.device("cpu"));
        if isinstance(val,torch.optim.Optimizer):
            logging.info("is a optim cpu")
            for param in val.state.values():
            # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(torch.device("cpu"))
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(torch.device("cpu"))
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(torch.device("cpu"))
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(torch.device("cpu"))
        return val;
    def __setattr__(self, key, value):
        #Trap the calls to ALL my object variables here itself.
        if (key in ('__dbcObj__', '__tDataColumns__')):
            return super().__setattr__(key, value);
        try:
            #logging.debug("DBCWrap: setattr : current known tabular data objects : {}".format(self.__tDataColumns__.keys()));

            #Convert numpy matrix back to TabularData object
            #Transpose the matrix to fit the internal form of TabularData objects.
            value = value.T;
            if(not value.flags['C_CONTIGUOUS']): #If the matrix is not C_CONTIGUOUS, make a copy in C_CONTGUOUS form.
                value = np.copy(value, order='C');
            #Build a new TabularData object using virtual transformation.
            #logging.debug("DBCWrap, setting : item {}, shape {}".format(key, value.shape));
            if key in self.__tDataColumns__:
                tDataCols = self.__tDataColumns__[key];
            #If we got to this line, then it means "key" was a TabularData object.
            # So we need to build a new TabularData object using the original column metadata.
                valueDF = DBC._dataFrameClass_._virtualData_(lambda:value, cols=tuple(tDataCols.keys()), colmeta=tDataCols, dbc=self.__dbcObj__);
            else:
            #If we go to this line, then it means "key" is a new variable.
            # So we need to build a new TabularData from scratch
                valueDF = DBC._dataFrameClass_._virtualData_(lambda:value, dbc=self.__dbcObj__);
            setattr(self.__dbcObj__, key, valueDF);
            return;
        except :
            logging.exception("DBCWrap : Exception ");
            pass;
        setattr(self.__dbcObj__, key, value);
