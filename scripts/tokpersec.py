import re
import matplotlib.pyplot as plt

# Function to extract step and tok/sec from the last 5 lines of a file
def extract_data(filepath):
    steps = []
    tok_sec = []
    with open(filepath, 'r') as file:
        lines = file.readlines()[-5:]  # Get the last 5 lines
        for line in lines:
            match = re.search(r"step\s+(\d+).*tok/sec:\s+([\d.]+)", line)
            if match:
                steps.append(int(match.group(1)))
                tok_sec.append(float(match.group(2)))
    return steps, tok_sec

# Filepaths
files = {
    "Baseline": "/home/min/a/kadhitha/scratch-space/workspace/chatgpt2-rotary-embed/logs/baseline.txt",
    "CUDA1": "/home/min/a/kadhitha/scratch-space/workspace/chatgpt2-rotary-embed/logs/cuda1.txt",
    "CUDA2": "/home/min/a/kadhitha/scratch-space/workspace/chatgpt2-rotary-embed/logs/cuda2.txt",
    "CUDA3": "/home/min/a/kadhitha/scratch-space/workspace/chatgpt2-rotary-embed/logs/cuda3.txt",
    "Triton": "/home/min/a/kadhitha/scratch-space/workspace/chatgpt2-rotary-embed/logs/triton.txt",
}

# Extract data from each file
data = {}
for label, filepath in files.items():
    # print(f"Processing {label}...")
    steps, tok_sec = extract_data(filepath)
    # print median of tok_sec with the last part in filepath
    tok_sec_median = sorted(tok_sec)[len(tok_sec) // 2]
    print(f"Median tok/sec for {filepath.split('/')[-1]}: {tok_sec_median:.2f}")
    data[label] = {"steps": steps, "tok_sec": tok_sec, "median": tok_sec_median}
    
# divide median values by baseline median value
baseline_median = data["Baseline"]["median"]
for label in data:
    data[label]["speedup"] = data[label]["median"] / baseline_median
    
print("Speedup values:")
for label in data:
    print(f"{label}: {data[label]['speedup']:.4f}x")

# Plotting
plt.figure(figsize=(10, 6))
for label, values in data.items():
    plt.plot(values["steps"], values["tok_sec"], marker="o", label=label)

plt.title("Token/sec during Training @ Step", fontsize=20)
plt.xlabel("Step", fontsize=18)
plt.ylabel("Token/sec", fontsize=18)
plt.legend(fontsize=16)
# set x ticks as 50, 51, 52, 53, 54
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim(50, 54)
# set x tick labels as 50, 51, 52, 53, 54
plt.xticks([50, 51, 52, 53, 54], ['50', '51', '52', '53', '54'])
# plt.grid(True)
plt.tight_layout()
plt.savefig("images/tokpersec.png")