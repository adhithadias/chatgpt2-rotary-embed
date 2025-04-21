import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Set the coordinate limits
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Set ticks to show -1, 1, 0.5, and 1.5
ax.set_xticks([-2.0, -1.5, -1, -0.5, 0.5, 1, 1.5, 2.0])
ax.set_yticks([-2.0, -1.5, -1, -0.5, 0.5, 1, 1.5, 2.0])

# Hide labels for 0.5 and 1.5
ax.set_xticklabels(['', '', '-1', '', '', '1', '', ''])
ax.set_yticklabels(['', '', '-1', '', '', '1', '', ''])

ax.annotate("0", xy=(1, 1), xytext=(-0.15, -0.225))

# Remove boundary lines and move axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('data', 0))  # Move x-axis to y=0
ax.spines['left'].set_position(('data', 0))    # Move y-axis to x=0

# Draw Cartesian axes
ax.axhline(0, color='black', linewidth=1)  # x-axis
ax.axvline(0, color='black', linewidth=1)  # y-axis

# Draw arrows (example)
# plt.arrow(x_start, y_start, dx, dy, ...)
ax.arrow(0, 0, 1, 1, head_width=0.15, head_length=0.2, linewidth=2.5, alpha=0.5, fc='blue', ec='blue', length_includes_head=True)
ax.arrow(0, 0, -1, 1, head_width=0.15, head_length=0.2, linewidth=2.5, alpha=1.0, fc='green', ec='green', length_includes_head=True)

# ax.annotate("", xy=(-1, 1), xytext=(0, 0),
#             arrowprops=dict(color='green', lw=1))
# ax.annotate("asdf", xy=(1, 1), xytext=(0, 0))

# Add labels
# ax.text(4, 3, 'Vector A', fontsize=12, color='blue')
# ax.text(-3, 5, 'Vector B', fontsize=12, color='green')

# Make the grid and aspect ratio nice
# ax.grid(True)
ax.set_aspect('equal')

# Adjust layout to remove extra spacing
plt.tight_layout()

# Save as image
plt.savefig("images/cartesian_arrows.png", dpi=300)
# plt.show()