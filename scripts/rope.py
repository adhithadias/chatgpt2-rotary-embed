import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
   
def plot_rope():
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
    
    ax.set_aspect('equal')

    # Adjust layout to remove extra spacing but keep some space for the arrows
    plt.tight_layout(pad=1.5)
    
    initial_vector = np.array([1, 1])
    
    x = initial_vector[0]
    y = initial_vector[1]
    ax.arrow(0, 0, x, y, head_width=0.15, head_length=0.2, linewidth=2.5, alpha=0.5, fc='blue', ec='blue', length_includes_head=True)
    ax.annotate(f"x", xy=(x, y), xytext=(x - 0.1, y + 0.1), fontsize=12)
        
    i = 1
    theta = 10000**(-2*(i-1)/2)

    # Draw arrows (example)
    # plt.arrow(x_start, y_start, dx, dy, ...)
    def animate(i):
        arrows = []
        l = max(i-1, 0)
        for pos in range(l, i):
            rotation_matrix = np.array([[np.cos(pos*theta), -np.sin(pos*theta)],
                                [np.sin(pos*theta), np.cos(pos*theta)]])
            # print(pos, np.sin(pos), np.cos(pos))
            
            # rotate the initial vector
            rotated_vector = np.dot(rotation_matrix, initial_vector)
            x = rotated_vector[0]
            y = rotated_vector[1]
            arrow = ax.arrow(0, 0, x, y, head_width=0.15, head_length=0.2, linewidth=2.5, alpha=0.5, fc='red', ec='red', length_includes_head=True)
            annot = ax.annotate(f"pos={pos}", xy=(x, y), xytext=(x + 0.1, y + 0.1), fontsize=12)
            
            arrows.append(arrow)
            arrows.append(annot)
        return arrows
    
    ani = animation.FuncAnimation(fig, animate, frames=8, interval=1000, blit=True)
    
    # save animation as gif
    ani.save(f'images/rope.gif', writer='pillow', fps=1)
    # plt.show()

    # ax.annotate("", xy=(-1, 1), xytext=(0, 0),
    #             arrowprops=dict(color='green', lw=1))
    # ax.annotate("asdf", xy=(1, 1), xytext=(0, 0))

    # Add labels
    # ax.text(4, 3, 'Vector A', fontsize=12, color='blue')
    # ax.text(-3, 5, 'Vector B', fontsize=12, color='green')

    # Make the grid and aspect ratio nice
    # ax.grid(True)


    # Save as image
    # plt.savefig(f"images/pos_in_polar_{add_vector}.png", dpi=300)
    
    # clear previous 7 plot commands
    
plot_rope()

