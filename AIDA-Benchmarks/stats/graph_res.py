import matplotlib.pyplot as plt
import pandas as pd

def formDataFrame(filename):

    data = pd.read_csv(filename)

    df = pd.DataFrame(data)

    # Convert the timestamps from microseconds to milliseconds (1 ms = 1000 µs)
    df['timestamp'] = df['Time (seconds)']

    return df


bench = "YCSB"
subject = "Average latency"
key = 'Average Latency (millisecond)'
#key = '99th Percentile Latency (millisecond)'

df_on = formDataFrame("ycsb_on_300.csv")
df_off = formDataFrame("ycsb_off_300.csv")

fig, ax1 = plt.subplots(figsize=(16, 8))


# Plot latency time series
#ax1.plot(df_on['timestamp'], df_on['latency'], label="Latency", color='blue')
ax1.plot(df_on['timestamp'], df_on[key], linewidth=2, label=f"{subject} (pg_stats ON)", color='red')

#ax1.plot(df_off['timestamp'], df_off['latency'], linestyle=':', label="Latency", color='blue')
ax1.plot(df_off['timestamp'], df_off[key], linestyle=':', linewidth=2, label=f"{subject} (pg_stats OFF)", color='blue')

plt.xlabel('Time (s)', fontsize=18, labelpad=20)  
plt.ylabel('Latency (ms)', fontsize=18)

plt.subplots_adjust(bottom=0.25)  # Increase bottom margin


fig.text(0.5, 0.02, f"Average with stats OFF : {df_off[key].mean().round(2)} ms", ha='center', fontsize=20)
fig.text(0.5, 0.07, f"Average with stats ON :  {df_on[key].mean().round(2)} ms", ha='center', fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.title(f"{subject}, stats on and off - {bench} Benchmark", fontsize=22)


# Show the plot
#ax1.legend()
ax1.grid()
plt.legend(fontsize=18)

plt.savefig(f"graphs/{bench}{subject}.pdf", format="pdf")
#plt.show()