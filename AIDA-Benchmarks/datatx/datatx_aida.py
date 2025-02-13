import argparse
import time;

import pandas as pd;

from aida.aida import *;
from aidacommon.aidaConfig import AConfig;

parser = argparse.ArgumentParser(description="Compute the data transfer speeds for AIDA channel");
parser.add_argument('--host', required=True);
parser.add_argument('--port', metavar='int', type=int, required=True);
parser.add_argument('--dbname', required=True);
parser.add_argument('--table', required=True);
parser.add_argument('--columns', nargs='+');
parser.add_argument('--iterate', default=5, metavar='int', type=int);
parser.add_argument('--ntwk', required=True);

args = parser.parse_args();
#print(args);

dw = AIDA.connect(host=args.host, dbname=args.dbname, user=args.dbname, passwd=args.dbname, jobName=__file__, port=args.port);

channel = AConfig.NTWKCHANNEL.__name__.split('.')[-1]

def timeload(td):
    st = time.time();
    data = pd.DataFrame(td.cdata);
    et = time.time();
    return(data, et-st);

def process(td, col='*'):
    for i in range(1, args.iterate+1):
        (data, ts) = timeload(td);
        print("aida,{},{},{},{},{},{}".format(args.ntwk, channel, i, args.table, col, ts));

td = getattr(dw, args.table)
if (args.columns is not None):
    for col in args.columns:
        process(td.project(col), col=col);
else:
    process(td);

