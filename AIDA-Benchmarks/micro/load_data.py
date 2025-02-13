import sys;
import time;
import timeit;

from micro.config import *;

jobName=thisJobName(__file__);


if(len(sys.argv) < 3):
    print("USAGE {} loadtype tablename".format(__file__));
    print("loadtype should be either AIDA for AIDA-DMRO or NumPy for NumPy array, pandas for pandas DF via external connection. NumpyOpt and pandasOpt for optimized database connection.");
    sys.exit(1);

loadType = sys.argv[1];
tableName = sys.argv[2];

if(loadType == "NumPy"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    for i in range(0, repeat):
        st = time.time();
        arr = getNumPy(con, tableName);
        et = time.time();
        print("{4},{3},{0},{1},{2:.3f}".format(loadType, tableName, et-st, i, host))

elif(loadType == "NumPyOpt"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    for i in range(0, repeat):
        st = time.time();
        arr = getNumPyOpt(con, tableName);
        et = time.time();
        print("{4},{3},{0},{1},{2:.3f}".format(loadType, tableName, et-st, i, host))

elif(loadType == "pandas"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    for i in range(0, repeat):
        st = time.time();
        arr = getPandasDF(con, tableName);
        et = time.time();
        print("{4},{3},{0},{1},{2:.3f}".format(loadType, tableName, et-st, i, host))

elif(loadType == "pandasOpt"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    for i in range(0, repeat):
        st = time.time();
        arr = getPandasDFOpt(con, tableName);
        et = time.time();
        print("{4},{3},{0},{1},{2:.3f}".format(loadType, tableName, et-st, i, host))

elif(loadType == 'AIDA'):
    dbc = getDBC(jobName);
    for i in range(0, repeat):
        st = time.time();
        tbl = getTabularData(dbc, tableName);
        et = time.time();
        print("{4},{3},{0},{1},{2:.3f}".format(loadType, tableName, et-st, i, host))

elif(loadType == 'AIDA-Matrix'):
    dbc = getDBC(jobName);
    for i in range(0, repeat):
        st = time.time();
        tbl = getTabularDataMatrix(dbc, tableName);
        et = time.time();
        print("{4},{3},{0},{1},{2:.3f}".format(loadType, tableName, et-st, i, host))




