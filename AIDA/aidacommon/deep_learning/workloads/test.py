import psutil
import time
import threading
import os
def measure():
    while(True):
#        psutil.cpu_percent()
         time.sleep(1)
         print(os.system('mpstat -P ALL')[1])
#        print("cpu util"+str(psutil.cpu_percent()))


checkLengthThread = threading.Thread(target=measure,args=());
#checkLengthThread.daemon = True;
checkLengthThread.start();
