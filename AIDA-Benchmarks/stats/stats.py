from aida.aida import *
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import requests

host = 'localhost'
dbname = 'benchbase'
user = 'test01'
passwd = 'password'
jobName = 'job1'
port = 55660

# # URL of your Prometheus server's API
# PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# # Function to query Prometheus for the up metric
# def get_up_metric():
#     query = 'up{job="your-job-name"}'  # Change to your job name
#     response = requests.get(PROMETHEUS_URL, params={'query': query})
#     data = response.json()
    
#     # Extract the value of the 'up' metric
#     if data['status'] == 'success' and 'data' in data:
#         return int(data['data']['result'][0]['value'][1])  # Return the value (1 or 0)
#     return 0

# # Initial value of the 'up' metric
# previous_value = get_up_metric()

dw = AIDA.connect(host,dbname,user,passwd,jobName,port)

x = []
y_response = []
y_throughput = []

print("Waiting...")
hit = 0
while hit < 30:
    val = dw._getResponseTime()
    if None == val:
        hit = 0
    else:
        hit += 1
    time.sleep(0.5)

print("Starting measurement.")

i = 0
responseTime = -1

while responseTime != 0:
    i += 1

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
    print( "Throughput : " + str(throughput) )

    time.sleep( 1 )
    
plt.plot(x, y_response, label='Response time per second')
plt.title("TPCH Response Time (no exporter)")
plt.xlabel("Time (s)")
plt.ylabel("Response time")
plt.legend()
plt.grid()
#plt.show()
plt.savefig("response_time.png")

plt.plot(x, y_response, label='Query throughput per second')
plt.title("TPCH Throughput (no exporter)")
plt.xlabel("Time (s)")
plt.ylabel("Throughput (qps)")
plt.legend()
plt.grid()
#plt.show()
plt.savefig("throughput.png")

dw._close()

    



