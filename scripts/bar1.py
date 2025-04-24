import matplotlib.pyplot as plt

# Your data
# without synchronization
# data = {
#     'layer-norm': 1137788233,
#     'attention': 7119247201,
#     'mlp': 2923400168,
# }

# with synchronization
data = {
    'layer-norm': 2696229220,
    'attention': 18721309402,
    'mlp': 16649267021,
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
plt.savefig("images/stacked_bar_chart1.png", bbox_inches='tight')