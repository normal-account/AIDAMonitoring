import argparse
import time;

import pandas.io.sql as psql;
import pymonetdb.sql;


parser = argparse.ArgumentParser(description="Compute the data transfer speeds for pymonetdb");
parser.add_argument('--host', required=True);
parser.add_argument('--port', metavar='int', type=int, required=True);
parser.add_argument('--dbname', required=True);
parser.add_argument('--table', required=True);
parser.add_argument('--columns', nargs='+');
parser.add_argument('--iterate', default=5, metavar='int', type=int);
parser.add_argument('--ntwk', required=True);
parser.add_argument('--buffersize', metavar='int', type=int);

args = parser.parse_args();


con = pymonetdb.Connection(args.dbname,hostname=args.host,port=args.port,username=args.dbname,password=args.dbname,autocommit=True);#
#increased buffer size for throughput.
optimized=False
if(args.buffersize is not None):
    con.replysize = args.buffersize;
    optimized=True;


def timeload(sql):
    st = time.time()
    data = psql.read_sql_query(sql=sql, con=con);
    et = time.time()
    return(data, et-st)


def process(sql, col='*'):
    for i in range(1, args.iterate+1):
        (data, ts) = timeload(sql);
        print("pymonetdb,{},{},{},{},{},{}".format(args.ntwk, optimized, i, args.table, col, ts));

sql='SELECT {} FROM {}';
if (args.columns is not None):
    for col in args.columns:
        process(sql.format(col, args.table), col=col);
else:
    process(sql.format('*', args.table));

