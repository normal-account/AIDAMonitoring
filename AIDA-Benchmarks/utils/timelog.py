import time;
import ntpath;

class TimeLog:
    def __init__(self, sourcename, logfile):
        self.sourcename = ntpath.basename(sourcename);
        self.logfile = open(logfile, 'a');
        self.prevtime = None;
        self.prevtag = None;

    def log(self, currenttag):
        t = time.time();
        if(self.prevtime):
            print("{},{},{}".format(self.sourcename, self.prevtag, t - self.prevtime), file=self.logfile);
        self.prevtime = t;
        self.prevtag = currenttag;

    def __del__(self):
        self.logfile.close();
