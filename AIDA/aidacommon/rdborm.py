import copyreg;

from aidacommon.dborm import *;
import aidacommon.rop;

class TabularDataRemoteStub(aidacommon.rop.RObjStub, TabularData):

    @aidacommon.rop.RObjStub.RemoteMethod()
    def filter(self, *selcols): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def join(self, otherTable, src1joincols, src2joincols, cols1=COL.NONE, cols2=COL.NONE, join=JOIN.INNER): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def aggregate(self, projcols, groupcols=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def agg(self, projcols, groupcols=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def project(self, projcols): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def order(self, orderlist): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def distinct(self): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def loadData(self, matrix=False): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __add__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __radd__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __mul__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __rmul__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __sub__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __rsub__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __truediv__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __rtruediv__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __pow__(self, power, modulo=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __matmul__(self, other): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __rmatmul__(self, other): pass;

    @property
    @aidacommon.rop.RObjStub.RemoteMethod()
    def T(self): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def __getitem__(self, item): pass;

    #WARNING !! Permanently disabled  !
    #Weakref proxy invokes this function for some reason, which is forcing the TabularData objects to materialize.
    #@aidacommon.rop.RObjStub.RemoteMethod()
    #def __len__(self): pass;

    @property
    @aidacommon.rop.RObjStub.RemoteMethod()
    def shape(self): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def vstack(self, othersrclist): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def hstack(self, othersrclist, colprefixlist=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def describe(self): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def sum(self, collist=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def avg(self, collist=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def count(self, collist=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def countd(self, collist=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def countn(self, collist=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def max(self, collist=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def min(self, collist=None): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def head(self, n=5): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def tail(self, n=5): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _U(self, func, *args, **kwargs): pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _genSQL_(self, *args, **kwargs): pass;

    @property
    @aidacommon.rop.RObjStub.RemoteMethod(compressResults=True)
    def cdata(self): pass;


copyreg.pickle(TabularDataRemoteStub, TabularDataRemoteStub.serializeObj);

class LogisticRegressionModelRemoteStub(aidacommon.rop.RObjStub):

    @aidacommon.rop.RObjStub.RemoteMethod()
    def get_model(self):
        pass;
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def decision_function(self,X):
        pass;
 
    @aidacommon.rop.RObjStub.RemoteMethod()
    def fit(self,X,y,sample_weight=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict_log_proba(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict_proba(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def score(self,X,y,sample_weight=None):
        pass;

copyreg.pickle(LogisticRegressionModelRemoteStub, LogisticRegressionModelRemoteStub.serializeObj);


class LinearRegressionModelRemoteStub(aidacommon.rop.RObjStub):

    @aidacommon.rop.RObjStub.RemoteMethod()
    def get_model(self):
        pass;
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def fit(self,X,y,sample_weight=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def get_params(self,deep=True):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def score(self,X,y,sample_weight=None):
        pass;
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def set_params(self,**params):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def coef(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def rank(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def singular(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def intercept(self):
        pass;
 
copyreg.pickle(LinearRegressionModelRemoteStub, LinearRegressionModelRemoteStub.serializeObj);

class DecisionTreeModelRemoteStub(aidacommon.rop.RObjStub):
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def apply(self,X,check_input=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def cost_complexity_pruning_path(self,X,y,sample_weight=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def decision_path(self,X,check_input=True):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def fit(self,X,y,sample_weight=None,check_input=True,X_idx_sorted=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict(self,X,check_input=True):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict_log_proba(self,X):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def predict_proba(self,X,check_input=True):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def score(self,X,y,sample_weight=None):
        pass;

copyreg.pickle(DecisionTreeModelRemoteStub,DecisionTreeModelRemoteStub.serializeObj);

class HelloWorldRemoteStub(aidacommon.rop.RObjStub):
    @aidacommon.rop.RObjStub.RemoteMethod()
    def _helloWorld(self):
        pass;
copyreg.pickle(HelloWorldRemoteStub, HelloWorldRemoteStub.serializeObj);

class DBCRemoteStub(aidacommon.rop.RObjStub):
    
    class ModelRepositoryRemoteStub(aidacommon.rop.RObjStub):
        def __getattribute__(self,item):
            try:
                return object.__getattribute__(self,item)
            except:
                pass
            result = super().__getattribute__(item)

#            if (isinstance(result,aidacommon.rop.RObjStub)):
 #               self._registerProxy_(item,result.proxyid);
  #              super().__setattr__(item, result);
            return result

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _tables(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _getDBTable(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _executeQry(self, sql, resultFormat='column'):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _Page(self, func, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _Plot(self, func, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _X(self, func, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _Schedule(self,iter_func,cond_func,test_func,*args,**kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _XP(self, func, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _GPU(self, func, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _extract_X(self, *args, **kwargs): pass
 
    @aidacommon.rop.RObjStub.RemoteMethod()
    def _extract_y(self, *args, **kwargs): pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _helloWorld(self):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _linearRegression(self, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _logisticRegression(self, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _decisionTree(self,*args,**kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _save(self,model_name,model,update=False):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _load(self,model_name):
        pass 

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _getBufferHitRate(self):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _getResponseTime(self):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _getGPUUsage(self):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _getCPUUsage(self):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _getThroughput(self):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _resetPgStatStatements(self):
        pass

    # @aidacommon.rop.RObjStub.RemoteMethod()
    # def _saveTorchModel(self, model_name, model, update=False):
    #     pass
    #
    # @aidacommon.rop.RObjStub.RemoteMethod()
    # def _loadTorchModel(self, model_name):
    #     pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _sql(self,sql):
        pass
    
    @aidacommon.rop.RObjStub.RemoteMethod()
    def _L(self, func, *args, **kwargs):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _ones(self, shape, cols=None):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _rand(self, shape, cols=None):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _randn(self, shape, cols=None):
        pass

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _toTable(self, tblrData, tableName=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _saveTblrData(self, tblrData, tableName, dbName=None, drop=False):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _dropTable(self, tableName, dbName=None):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _close(self):
        pass;

    @aidacommon.rop.RObjStub.RemoteMethod()
    def _registerProxy_(self, attrname, proxyid):
        pass;

    #TODO: write corresponding code in the DBC class
    @aidacommon.rop.RObjStub.RemoteMethod()
    def _setattr_(self, key, value, returnAttr=False):
        pass;

    def __getattribute__(self, item):
        try:
            #Check if we have the attribute locally.
            return object.__getattribute__(self, item);
        except:
            pass;

        #Find the object remotely ...
        result = super().__getattribute__(item);

        #If this a stub object, we are going to set it locally and also listen for updates on it.
        if(isinstance(result, aidacommon.rop.RObjStub)):
            self._registerProxy_(item, result.proxyid);
            super().__setattr__(item, result);
        #return the attribute.
        return result

    def __setattr__(self, key, value):
        if(key.startswith('_')):
            super().__setattr__(key, value);
        else:
            curval = None;
            try:
                #Find if there is an existing object for this key locally/remotely.
                curval = self.__getattribute__(key);
            except:
                pass;

            if(curval):
                #If there is currently an object which is a remote object stub,
                if(isinstance(curval, aidacommon.rop.RObjStub)):
                    # but the new one is not, we cannot allow this.
                    if(not isinstance(value, aidacommon.rop.RObjStub)):
                        raise AttributeError("Error: cannot replace a remote stub with a regular object")
                    #If we are replacing one remote obj stub with another, they need to have compatible stubs.
                    if(not (isinstance(value, curval.__class__))):
                        raise AttributeError("Error: the remote stubs are not compatible {} {}".format(value.__class__, curval.__class__))
                #TODO if curval and value are the same, do nothing.

            #Ask the remote object to set this attribute. it can also return a stub if this a stub and we do not have
            #currently a stub pointing to it.
            #TODO ... do we need to get stub back from do this ? why cannot we just use the existing stub ?
            robj = self._setattr_(key, value, not(curval) and isinstance(value, aidacommon.rop.RObjStub) );
            if(robj): #The remote object returned a stub.
                self._registerProxy_(key, robj.proxyid);
                super().__setattr__(key, robj);

    #TODO: trap __del__ and call _close ?

copyreg.pickle(DBCRemoteStub.ModelRepositoryRemoteStub,DBCRemoteStub.ModelRepositoryRemoteStub.serializeObj);
copyreg.pickle(DBCRemoteStub, DBCRemoteStub.serializeObj);
