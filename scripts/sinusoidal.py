import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate time values
t = np.linspace(0, 2 * np.pi, 100)  # From 0 to 2*pi (one cycle), with 100 points


def plot_cosine(t):
    yc = np.cos(t)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, yc, label='')
    # plt.xlabel('Position (pos)', fontsize=14)
    # plt.ylabel('sin(t)')
    # plt.title(r'$PE(pos, 2i) = \cos\left(t/10000^{2i/d}\right), 0 \leq i \leq d/2$', fontsize=16)  # Example with exponential
    ax.set_yticks([-1, 0, 1], ['-1', '0', '1'])
    ax.set_xticks([1, 2, 3, 4, 5, 6], ['1', '2', '3', '4', '5', '6'])
    
    ax.spines['bottom'].set_position(('data', 0))  # Move x-axis to y=0
    ax.spines['left'].set_position(('data', 0))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # add x-axis label at the end named pos
    ax.annotate("pos", xy=(6, 0), xytext=(6.5, -0.2), fontsize=12)
    plt.tight_layout()
    plt.savefig('images/cosine_function.png')
    
    
    for pos in range(0, 7):
        plt.plot(pos, np.cos(pos), 'ro', markersize=6)
    
    plt.tight_layout()
    plt.savefig('images/cosine_function_marked.png')
    plt.close()
    
    
def plot_sine(t):
    yc = np.sin(t)
    fig, ax = plt.subplots(figsize=(8, 4))
    # plt.figure(figsize=(8, 4))  # Set the figure size
    ax.plot(t, yc, label='')
    # plt.xlabel('Position (pos)', fontsize=14)
    # plt.ylabel('sin(t)')
    # plt.title(r'$PE(pos, 2i) = \sin\left(t/10000^{2i/d}\right), 0 \leq i \leq d/2$', fontsize=16)
    ax.set_yticks([-1, 0, 1], ['-1', '0', '1'])
    ax.set_xticks([1, 2, 3, 4, 5, 6], ['1', '2', '3', '4', '5', '6'])
    
    ax.spines['bottom'].set_position(('data', 0))  # Move x-axis to y=0
    ax.spines['left'].set_position(('data', 0))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # add x-axis label at the end named pos
    ax.annotate("pos", xy=(6, 0), xytext=(6.5, -0.2), fontsize=12)
    
    plt.tight_layout()
    plt.savefig('images/sine_function.png')
    
    for pos in range(0, 7):
        ax.plot(pos, np.sin(pos), 'ro', markersize=6)
    
    plt.tight_layout()
    plt.savefig('images/sine_function_marked.png')
    plt.close()
    
def plot_cartesian(add_vector=True):
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
    
    if add_vector:
        x = 0.5
        y = 0.5
        ax.arrow(0, 0, x, y, head_width=0.15, head_length=0.2, linewidth=2.5, alpha=0.5, fc='blue', ec='blue', length_includes_head=True)
        ax.annotate(f"x", xy=(x, y), xytext=(x + 0.1, y + 0.1), fontsize=12)
        
    # Draw arrows (example)
    # plt.arrow(x_start, y_start, dx, dy, ...)
    def animate(i):
        arrows = []
        for pos in range(0, i):
            # print(pos, np.sin(pos), np.cos(pos))
            x = np.sin(pos)
            y = np.cos(pos)
            if (add_vector):
                x += 0.5
                y += 0.5
            arrow = ax.arrow(0, 0, x, y, head_width=0.15, head_length=0.2, linewidth=2.5, alpha=0.5, fc='red', ec='red', length_includes_head=True)
            annot = ax.annotate(f"pos={pos}", xy=(x, y), xytext=(x + 0.1, y + 0.1), fontsize=12)
            
            arrows.append(arrow)
            arrows.append(annot)
        return arrows
    
    ani = animation.FuncAnimation(fig, animate, frames=8, interval=1000, blit=True)
    
    # save animation as gif
    ani.save(f'images/cartesian_arrows_{add_vector}.gif', writer='pillow', fps=1)
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
    plt.savefig(f"images/pos_in_polar_{add_vector}.png", dpi=300)
    
    # clear previous 7 plot commands
    
plot_sine(t)
plot_cosine(t)
plot_cartesian(add_vector=False)
plot_cartesian(add_vector=True)

