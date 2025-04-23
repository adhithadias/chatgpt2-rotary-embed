import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = '/home/min/a/kadhitha/scratch-space/workspace/chatgpt2-rotary-embed/results/initial_bz.csv'
data = pd.read_csv(file_path)

# Split the 'config' column into separate columns for batch size (b) and sequence length (s)
data[['b', 's']] = data['config'].str.extract(r'b=(\d+)\|s=(\d+)')
data['b'] = data['b'].astype(int)
data['s'] = data['s'].astype(int)

data['FLOPs'] = 3 * data['b'] * data['s'] * 12 * 64

# Filter data for s=1024 and s=2048
data_1024 = data[data['s'] == 1024]
data_2048 = data[data['s'] == 2048]

# Function to plot grouped bar plots with speedup
def plot_grouped_bars_with_speedup(data, title, output_file, throughput : bool =False):
    labels = data['b'].astype(str)  # Batch sizes as labels
    default = data['default(ms)']
    cuda1 = data['cuda1(ms)']
    cuda2 = data['cuda2(ms)']
    FLOPs = data['FLOPs']
    
    if throughput:
        default = FLOPs / default * 10**3
        cuda1 = FLOPs / cuda1 * 10**3
        cuda2 = FLOPs / cuda2 * 10**3

    # Calculate speedup over default
    if throughput:
        speedup_cuda1 = cuda1 / default 
        speedup_cuda2 = cuda2 / default
        geomean_cuda1 = np.exp(np.mean(np.log(speedup_cuda1)))
        geomean_cuda2 = np.exp(np.mean(np.log(speedup_cuda2)))
    else:
        speedup_cuda1 = default / cuda1
        speedup_cuda2 = default / cuda2
        geomean_cuda1 = np.exp(np.mean(np.log(speedup_cuda1)))
        geomean_cuda2 = np.exp(np.mean(np.log(speedup_cuda2)))

    x = np.arange(len(labels))  # Label locations
    width = 0.25  # Width of the bars

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plots for execution times
    ax1.bar(x - width, default, width, label='Baseline (ms)', color='blue')
    ax1.bar(x, cuda1, width, label='CUDA1 (ms)', color='orange')
    ax1.bar(x + width, cuda2, width, label='CUDA2 (ms)', color='green')

    ax1.set_xlabel('Batch Size (b)', fontsize=18)
    if throughput:
        ax1.set_ylabel('Throughput (FLOPS)', fontsize=18)
    else:
        ax1.set_ylabel('Time (ms)', fontsize=18)
    ax1.set_title(f"{title}, Geomean Speedup: {geomean_cuda2:.2f}", fontsize=22)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=18)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.yaxis.get_offset_text().set_fontsize(16)
    ax1.legend(loc='upper left', fontsize=14)

    # Secondary y-axis for speedup
    ax2 = ax1.twinx()
    ax2.plot(x, speedup_cuda1, label='Speedup CUDA1', color='red', marker='o', linestyle='--')
    ax2.plot(x, speedup_cuda2, label='Speedup CUDA2', color='purple', marker='o', linestyle='--')
    ax2.set_ylabel('Speedup over Baseline', fontsize=18)
    ax2.legend(loc='upper right', fontsize=14)
    
    # limit secondary y-axis to 0.9 to 1.3
    ax2.set_ylim(0.9, 1.3)
    ax2.set_yticks(np.arange(0.9, 1.4, 0.1))
    
    # set y ticks fontsize to 18
    ax2.tick_params(axis='y', labelsize=16)
    
    # add a horizontal line at y=1.0
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Plot for s=1024
plot_grouped_bars_with_speedup(data_1024, 'Performance and Speedup \nSeqlen=1024', 'images/performance_speedup_s1024.png', throughput=True)

# Plot for s=2048
plot_grouped_bars_with_speedup(data_2048, 'Performance and Speedup \nSeqlen=2048', 'images/performance_speedup_s2048.png', throughput=True)