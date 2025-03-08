import matplotlib.pyplot as plt
import pandas as pd

# Load latency data (assuming CSV format)
data = pd.read_csv("latency.csv")  # Replace with your actual output file

df = pd.DataFrame(data)

# Convert the timestamps from microseconds to milliseconds (1 ms = 1000 µs)
df['timestamp'] = df['Start Time (microseconds)'] / 1000
# Calculate the offset from the first timestamp (in milliseconds)
df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

# Plot latency time series
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], data['Latency (microseconds)'], label="Latency", color='blue')

# Label the axes
plt.xlabel("Timestamp")
plt.ylabel("Latency (ms)")
plt.title("Latency Time Series - TPC-H Benchmark")

# Show the plot
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()