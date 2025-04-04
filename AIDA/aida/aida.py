from abc import ABCMeta
import socket;

import pickle;
from tblib import pickling_support;
pickling_support.install();
import dill as custompickle;

import aidacommon.aidaConfig;

from aidacommon.dborm import *;

import aidas.dmro as dmro;
import aidas.aidas as aidas;

#import logging;
#logging.basicConfig(filename='aida.log', level=logging.INFO);

class AIDA(metaclass=ABCMeta):
    myvar = 420

    @staticmethod
    def connect (host, dbname, user, passwd, jobName=None, port=55660):

        aidacommon.aidaConfig.loadConfig(topic='AIDACLIENT');

        # Make network connection.
        sock = socket.socket();
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1);
        (host,port) = aidacommon.aidaConfig.portMapper(host,port); #See if this port is tunnelled.
        sock.connect((host, port));

        #File handles to work directly with the custompickle functions.
        wf = sock.makefile('wb'); rf = sock.makefile('rb');

        #send authentication info to the connection manager running at the database.
        custompickle.dump((dbname, user, passwd, jobName),wf); wf.flush();

        ret = custompickle.load(rf);
        #Handsake to let the other side know that we have established the stubs.
        custompickle.dump(None,wf); wf.flush();

        #TODO check ret and throw it as an exception in case of some error.
        return ret;

    @staticmethod
    def callback(arg):
        # Initialize the DMRO repository.
        try:
            dmro.DMROrepository('aidasys');
        except Exception as e:
            pass

        import aidasys;

        #aidasys.increment()

        return aidasys.memtest

    @staticmethod
    def udf(arg):
        import aidas.bootstrap;
        aidas.bootstrap.bootstrap();
        
        import queue;
        requestQueue = queue.Queue();
        resultQueue = queue.Queue();
        memtest = 500

        import aidasys;
        aidasys.requestQueue = requestQueue;
        aidasys.resultQueue = resultQueue;
        aidasys.memtest = memtest
        conMgr = aidasys.conMgr
        
        while True:
            try:
                (jobName, request) = requestQueue.get();
                dbcObj = conMgr.get(jobName);
                dbcObj._executeRequest(plpy,request);
                requestQueue.task_done()
            except Exception as e:
                logging.info("EXCEPTION : " + str(e));
                break;
        
        return 1

def head(tdata, n=5):
    print(tdata.head(n));

def tail(tdata, n=5):
    print(tdata.tail(n));

def describe(tdata):
    print(tdata.describe());

def tables(dw):
    print(dw._tables());

try:
    #from IPython.core.display import display;
    from IPython.display import display;
    from IPython.display import IFrame;
    from PIL import Image;
    from urllib.parse import urlparse, urlunparse

    def show(resource, width='100%', height=500):
        if(isinstance(resource, str)): #IF this is a URL
            url = urlparse(resource);
            (host, port) = (url.hostname, 80 if(url.port is None) else url.port);
            (host, port) = aidacommon.aidaConfig.portMapper(host, port)
            resource = urlunparse((url.scheme, host+ ('' if(port==80) else ':'+str(port)), url.path, url.params, url.query, url.fragment))
            display(IFrame(src=resource, width=width, height=height));
        else:
            display(Image.open(resource));
except:
    pass;
