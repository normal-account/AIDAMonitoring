import sys;
import time;
import timeit;


from micro.config import *;

jobName=thisJobName(__file__);


if(len(sys.argv) < 5):
    print("USAGE {} type tablename table2Name numcolumns".format(__file__));
    print("type should be either AIDA or AIDA-Matrix or pandasOpt for pandas DF via external connection");
    sys.exit(1);

loadType = sys.argv[1];
tableName = sys.argv[2];
table2Name = sys.argv[3];
numColumns=int(sys.argv[4]);

#The columns over which join will be done.
ljoincols=[];
rjoincols=[];
for i in range(0, numColumns):
    ljoincols.append("c{}".format(i));
    rjoincols.append("b{}".format(i));
ljoincols=tuple(ljoincols);
rjoincols=tuple(rjoincols);

if(loadType == "pandasOpt"):
    con = pymonetdb.Connection(dbname,hostname=host,username=user,password=passwd,autocommit=True);
    arr = getPandasDFOpt(con, tableName);
    arr2 = getPandasDFOpt(con, table2Name);
    for i in range(0, repeat+warmup):
        st = time.time();
        res = pandas.merge(arr, arr2, how='inner', left_on=ljoincols, right_on=rjoincols);
        et = time.time();
        numRows=len(res.index);
        print("{4},{3},{0},{1},{2:.7f},{5},{6}".format(loadType, table2Name, et-st, i, host, numRows, numColumns));

elif(loadType == "AIDA"):
    dbc = getDBC(jobName);
    tbl = dbc._getDBTable(tableName);
    tbl2 = dbc._getDBTable(table2Name);
    #tbl = getTabularDataMatrix(dbc, tableName);
    #tbl2 = getTabularDataMatrix(dbc, table2Name);
    for i in range(0, repeat+warmup):
        st = time.time();
        res = tbl.join(tbl2, ljoincols, rjoincols, COL.ALL, COL.ALL);
        #print(res.genSQL.sqlText)
        res.loadData(matrix=True);
        #res.loadData();
        et = time.time();
        numRows=res.numRows;
        print("{4},{3},{0},{1},{2:.7f},{5},{6}".format(loadType, table2Name, et-st, i, host,numRows, numColumns));

elif(loadType == "AIDA-Matrix"):
    dbc = getDBC(jobName);
    #tbl = getTabularDataMatrix(dbc, tableName).project(('c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10'));
    #tbl2 = getTabularDataMatrix(dbc, table2Name).project(('c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10'));
    tbl = ((getTabularDataMatrix(dbc, tableName)) * 1); tbl.loadData(matrix=True);
    tbl2 = ((getTabularDataMatrix(dbc, table2Name)) * 1); tbl2.loadData(matrix=True);
    for i in range(0, repeat+warmup):
        st = time.time();
        res = tbl.join(tbl2, ljoincols, rjoincols, COL.ALL, COL.ALL);
        #print(res.genSQL.sqlText)
        res.loadData(matrix=True);
        #res.loadData();
        et = time.time();
        numRows=res.numRows;
        print("{4},{3},{0},{1},{2:.7f},{5},{6}".format(loadType, table2Name, et-st, i, host,numRows, numColumns));

