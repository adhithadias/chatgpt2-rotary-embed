import matplotlib.pyplot as plt

# Your data
# without synchronization
# data = {
#     'token-emb': 132951436,
#     'transformer\nblocks': 11342064896,
#     'lm-final': 47300327,
#     'lm-head': 104392863,
#     'loss-calc': 98313079,
# }

# with synchronization
data = {
    'token-emb': 220063271,
    'transformer\nblocks': 38236447673,
    'lm-final': 113853844,
    'lm-head': 7531774067,
    'loss-calc': 8440770870,
}

labels = list(data.keys())
values = list(data.values())
total = sum(values)
percentages = [(v / total) * 100 for v in values]

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(4, 12))

bottom = 0
t = 0
for i, (label, value, percentage) in enumerate(zip(labels, values, percentages)):
    ax.bar(0, value, bottom=bottom, label=label)
    ax.text(0, t + (total/len(labels)) / 2, f'{label}\n({percentage:.1f}%)', ha='center', va='center', color='white', fontsize=28, fontweight='bold')
    bottom += value
    t += total/len(labels)
    

# Remove axes
ax.axis('off')

# Set y-limits to ensure the full bar is visible
ax.set_ylim(0, total)

# Show the plot
# plt.show()
plt.savefig("images/stacked_bar_chart3.png", bbox_inches='tight')