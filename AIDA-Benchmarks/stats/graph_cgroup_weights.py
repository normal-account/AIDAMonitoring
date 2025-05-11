import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def formDataFrame(filename):

    data = pd.read_csv(filename)

    df = pd.DataFrame(data)

     # Convert the timestamps from microseconds to milliseconds (1 ms = 1000 µs)
    df['timestamp'] = df['Start Time (microseconds)'] // 1

    df['Latency (microseconds)'] = df['Latency (microseconds)'] / 1000

    # Calculate the offset from the first timestamp (in milliseconds)
    df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

    df_agg = df.groupby('timestamp', as_index=False)['Latency (microseconds)'].mean()

    df_agg = df_agg[df_agg['timestamp'] > 20] 

    total_time_sec = (180)# Convert µs to seconds
    total_ops = len(df)

    throughput = total_ops / total_time_sec if total_time_sec > 0 else 0

    df_agg = df_agg[df_agg['timestamp'] > 20] 

    return df_agg, throughput


bench = "YCSB"
subject = "Latency"
key = 'Latency (microseconds)'
#key = '99th Percentile Latency (millisecond)'

df_off, t_off = formDataFrame("results/ycsb_gpu_max_16.csv")
df_max, t_max = formDataFrame("results/ycsb_gpu_weight_20_100.csv")

keyword_off = "barebones"
keyword_max = "w/ Torch Linear Regression at weight 20/100"

fig, ax1 = plt.subplots(figsize=(16, 9))

# Plot latency time series
plt.plot(df_max['timestamp'], df_max[key], linewidth=2, label=f"{subject} ({keyword_max})", color='red')

plt.plot(df_off['timestamp'], df_off[key], linestyle=':', linewidth=2, label=f"{subject} ({keyword_off})", color='blue')


plt.xlabel('Time (s)', fontsize=18, labelpad=20)  
plt.ylabel('Latency (ms)', fontsize=18)


data = [
    [f"{keyword_max}", f"{df_max[key].mean():.2f}", f"{df_max[key].quantile(0.95):.2f}", t_max],
    [f"{keyword_off}", f"{df_off[key].mean():.2f}", f"{df_off[key].quantile(0.95):.2f}", t_off],
]

# Column labels
columns = ['Measure', 'Mean latency (ms)', '95th Percentile latency (ms)', 'Throughput (rps)']
plt.subplots_adjust(bottom=0.3)  # Increase bottom margin

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.title(f"{subject} - {bench} Benchmark", fontsize=22)

ax1.grid()
plt.legend(fontsize=18)

table_ax = fig.add_axes([0.1, 0.15, 0.8, 0.2])  # Adjust y-position (0.1) and height (0.2)
table_ax.axis('off')  # Turn off the axes for the table

table = table_ax.table(cellText=data, colLabels=columns)

# Set the font size of the table and make the header bold
table.auto_set_font_size(False)
#table.set_fontsize(18)  # Increase the font size for the table

# Make the header bold
for (i, j), cell in table.get_celld().items():
    if j == 0:
        cell.set_width(0.4)  # Adjust the height of the rows
    if i == 0:  # Row 0 is the header row
        #cell.set_fontweight('bold')  # Set the header cells to bold
        cell.set_height(0.18)  # Adjust the height of the rows
        cell.set_text_props(weight='bold', fontsize=16)  # Bold and larger header font
    else:
        cell.set_height(0.15)  # Adjust the height of the rows
        cell.set_text_props(fontsize=16)  # Bold and larger header font

#fig.text(0.85, 0.02, f"[Average, 95th latency] with {keyword_off} : [{df_off[key].mean():.2f} ms, {df_off[key].quantile(0.95):.2f} ms]", ha='right', fontsize=20)
#fig.text(0.85, 0.07, f"[Average, 95th latency] with {keyword_on} : [{df_on[key].mean():.2f} ms, {df_on[key].quantile(0.95):.2f} ms]", ha='right', fontsize=20)
#fig.text(0.85, 0.12, f"[Average, 95th latency] with {keyword_on2} : [{df_on2[key].mean():.2f} ms, {df_on2[key].quantile(0.95):.2f} ms]", ha='right', fontsize=20)



# Show the plot
#ax1.legend()


plt.savefig(f"cgroup/{bench}{subject}_latency_weight.pdf", format="pdf")
plt.show()