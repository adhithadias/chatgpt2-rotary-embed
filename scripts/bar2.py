import matplotlib.pyplot as plt

# Your data
# without synchronization
# data = {
#     'qkv-proj': 1844238852,
#     'rope': 2486660208,
#     'sdpa': 1093775911,
#     'output-proj': 1228360462,
# }

# with synchronization
data = {
    'qkv-proj': 6482893539,
    'rope': 3592007773,
    'sdpa': 4025878544,
    'output-proj': 2836534332,
}

labels = list(data.keys())
values = list(data.values())
total = sum(values)
percentages = [(v / total) * 100 for v in values]

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(4, 12))

bottom = 0
for i, (label, value, percentage) in enumerate(zip(labels, values, percentages)):
    ax.bar(0, value, bottom=bottom, label=label)
    ax.text(0, bottom + value / 2, f'{label}\n({percentage:.1f}%)', ha='center', va='center', color='white', fontsize=28, fontweight='bold')
    bottom += value

# Remove axes
ax.axis('off')

# Set y-limits to ensure the full bar is visible
ax.set_ylim(0, total)

# Show the plot
# plt.show()
plt.savefig("images/stacked_bar_chart2.png", bbox_inches='tight')