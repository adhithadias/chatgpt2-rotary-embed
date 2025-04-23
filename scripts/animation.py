import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Set up the initial plot
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Animating Arrows")

# Define arrow data
arrow_data = [
    {'x': 1, 'y': 2, 'dx': 3, 'dy': 2, 'color': 'red'},
    {'x': 5, 'y': 1, 'dx': -2, 'dy': 4, 'color': 'blue'},
    {'x': 2, 'y': 7, 'dx': 4, 'dy': -1, 'color': 'green'}
]

# Create an empty list to store arrows
arrows = []

# Define the animation function
def animate(i):
    if i < len(arrow_data):
        arrow = ax.arrow(arrow_data[i]['x'], arrow_data[i]['y'],
                         arrow_data[i]['dx'], arrow_data[i]['dy'],
                         head_width=0.5, head_length=0.8, fc=arrow_data[i]['color'], ec=arrow_data[i]['color'])
        arrows.append(arrow)
    return arrows

# Create the animation object
ani = animation.FuncAnimation(fig, animate, frames=len(arrow_data) + 1, interval=1000, blit=True)

# Display the animation
plt.show()