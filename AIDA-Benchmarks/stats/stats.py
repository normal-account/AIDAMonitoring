from aida.aida import *
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import requests
import re
import threading

host = 'localhost'
dbname = 'benchbase'
user = 'test01'
passwd = 'password'
jobName = 'job1'
port = 55660

scrape_seconds = []
eval_seconds = []
stop_fetch = False

# URL of your Prometheus server's API
PROMETHEUS_URL = "http://localhost:9090/metrics"

# Function to query Prometheus for the up metric
def fetch_export_times():
    global scrape_seconds, eval_seconds, stop_fetch
    previous_value = 0

    start_time = time.time()

    while not stop_fetch:
        response = requests.get(PROMETHEUS_URL)
        if response.status_code == 200:

            metrics_text = response.text
            # Regex pattern to extract the specific metric value
            pattern = r'prometheus_target_interval_length_seconds_count\{interval="1s"\} (\d+)'
            match = re.search(pattern, metrics_text)

            if match:
                metric_value = int(match.group(1))  # Extract and convert to integer
                if previous_value != metric_value:
                    previous_value = metric_value
                    print(f"Metric Value: {metric_value}")

                    elapsed_seconds = time.time() - start_time

                    if metric_value % 2 == 0:
                        scrape_seconds.append(elapsed_seconds)
                    else:
                        eval_seconds.append( elapsed_seconds )

dw = AIDA.connect(host,dbname,user,passwd,jobName,port)

print( dw._getResponseTime() )

x = []
y_response = []
y_throughput = []

print("Waiting...")
hit = 0
while hit == 0:
    val = dw._getResponseTime()
    if None == val:
        hit = 0
    else:
        hit += 1
    time.sleep(0.5)

print("Starting measurement.")

# Start the monitoring thread
monitor_thread = threading.Thread(target=fetch_export_times, daemon=True)
monitor_thread.start()

i = 0
responseTime = -1

while responseTime != 0:
    i += 0.5

    # Getting response time
    responseTime = dw._getResponseTime()

    if responseTime == None:
        responseTime = 0

    y_response.append( responseTime )


    # Getting throughput
    throughput = dw._getThroughput()

    if throughput == None:
        throughput = 0

    y_throughput.append( throughput )


    # Resetting pg_stat_statements
    dw._resetPgStatStatements()

    x.append( i )

    print( "Response time : " + str(responseTime) )
    #print( "Throughput : " + str(throughput) )

    time.sleep( 0.5 )


# Wait for the thread to finish
stop_fetch = True
monitor_thread.join() 

x = x[10:]  # Slice x to exclude the first 10 points
y_response = y_response[10:]  # Slice y to exclude the first 10 points

scrape_seconds = scrape_seconds[1:]
eval_seconds = eval_seconds[1:]

print(scrape_seconds)
print("\n")
print(eval_seconds)

#print(x)

# plt.figure(figsize=(16, 6))  # Width = 10 inches, Height = 6 inches
# plt.plot(x, y_response, label='Response time per second')

# ymin, ymax = plt.ylim()
# plt.vlines(x=scrape_seconds, color='r', ymin=ymin, ymax=ymax, linestyle='dashed', label="Scrape")
# plt.vlines(x=eval_seconds, color='g', ymin=ymin, ymax=ymax, linestyle='dashed', label="Evaluation")

# plt.title("TPC-H Response Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Response time (ms)")
# plt.legend()
# plt.grid()
# plt.savefig("response_time_tpch.pdf")
# plt.show()

# plt.plot(x, y_response, label='Query throughput per second')
# plt.title("TPCH Throughput (no exporter)")
# plt.xlabel("Time (s)")
# plt.ylabel("Throughput (qps)")
# plt.legend()
# plt.grid()
# #plt.show()
# plt.savefig("throughput.png")

dw._close()

    



