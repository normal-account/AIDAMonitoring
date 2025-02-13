import sys;
import time;
import timeit;

from micro.config import *;

jobName=thisJobName(__file__);


if(len(sys.argv) < 3):
    print("USAGE {} loadtype tablename".format(__file__));
    print("loadtype should be either AIDA for AIDA-Matrix or NumPyOpt for NumPy array, pandasOpt for pandas DF via external connection");
    sys.exit(1);

loadType = sys.argv[1];
tableName = sys.argv[2];

if(loadType == "NumPyOpt"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    arr = getNumPyOpt(con, tableName);
    for i in range(0, warmup):
        res = arr * arr;
    st = time.time();
    for i in range(0, repeat):
        res = arr * arr;
    et = time.time();
    print("{4},{3},{0},{1},{2:.7f}".format(loadType, tableName, et-st, '-', host))

elif(loadType == "pandasOpt"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    arr = getPandasDFOpt(con, tableName);
    for i in range(0, warmup):
        res = arr * arr;
    st = time.time();
    for i in range(0, repeat):
        res = arr * arr;
    et = time.time();
    print("{4},{3},{0},{1},{2:.7f}".format(loadType, tableName, et-st, '-', host))

elif(loadType == "AIDA-Matrix"):
    dbc = getDBC(jobName);
    tbl = getTabularDataMatrix(dbc, tableName);
    for i in range(0, warmup):
        res = tbl * tbl;
    st = time.time();
    for i in range(0, repeat):
        res = tbl * tbl;
    et = time.time();
    print("{4},{3},{0},{1},{2:.7f}".format(loadType, tableName, et-st, '-', host))

elif(loadType == 'AIDA-RWS'):
    dbc = getDBC(jobName);
    tbl = getTabularDataMatrix(dbc, tableName);
    dbc.warmup = warmup;
    dbc.repeat = repeat;
    def myfunc(w, t):
        import time;
        ws = time.time();
        for i in range(0, w.warmup):
            res = t * t;
        we = time.time();
        w.wt = we-ws;
        for i in range(0, w.repeat):
            res = t * t;
        return res;

    st = time.time();
    res = dbc._X(myfunc, tbl);
    et = time.time();
    print("{4},{3},{0},{1},{2:.3f}".format(loadType, tableName, et-st-dbc.wt, '-', host))