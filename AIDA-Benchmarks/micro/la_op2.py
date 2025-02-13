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
table2Name = sys.argv[3];

if(loadType == "NumPyOpt"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    arr = getNumPyOpt(con, tableName);
    arr2 = getNumPyOpt(con, table2Name).T;
    for i in range(0, warmup):
        res = arr @ arr2;
    st = time.time();
    for i in range(0, repeat):
        res = arr @ arr2;
    et = time.time();
    print("{4},{3},{0},{1},{2:.7f}".format(loadType, tableName, et-st, '-', host))
    #print(res)

elif(loadType == "pandasOpt"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    arr = getPandasDFOpt(con, tableName);
    arr2 = getPandasDFOpt(con, table2Name).T;
    for i in range(0, warmup):
        res = arr.dot(arr2);
        #res = arr @ arr2;
    st = time.time();
    for i in range(0, repeat):
        res = arr.dot(arr2);
        #res = arr @ arr2;
    et = time.time();
    print("{4},{3},{0},{1},{2:.7f}".format(loadType, tableName, et-st, '-', host))
    #print(res)

elif(loadType == "AIDA-Matrix"):
    dbc = getDBC(jobName);
    tbl = getTabularDataMatrix(dbc, tableName);
    tbl2 = getTabularDataMatrix(dbc, table2Name).T;
    for i in range(0, warmup):
        res = tbl @ tbl2;
    st = time.time();
    for i in range(0, repeat):
        res = tbl @ tbl2;
    et = time.time();
    print("{4},{3},{0},{1},{2:.7f}".format(loadType, tableName, et-st, '-', host))
    #print(res.rows)

elif(loadType == 'AIDA-RWS'):
    dbc = getDBC(jobName);
    tbl = getTabularDataMatrix(dbc, tableName);
    tbl2 = getTabularDataMatrix(dbc, table2Name).T;
    dbc.warmup = warmup;
    dbc.repeat = repeat;
    def myfunc(w, t, t2):
        import time;
        ws = time.time();
        for i in range(0, w.warmup):
            res = t @ t2;
        we = time.time();
        w.wt = we-ws;
        for i in range(0, w.repeat):
            res = t @ t2;
        return res;

    st = time.time();
    res = dbc._X(myfunc, tbl, tbl2);
    et = time.time();
    print("{4},{3},{0},{1},{2:.3f}".format(loadType, tableName, et-st-dbc.wt, '-', host))
    #print(res.rows)
