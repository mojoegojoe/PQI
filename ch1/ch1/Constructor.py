
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import math

def generate_fibonacci(n):
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]
def draw_circle_with_fib(val):
    n = int(val)
    ax.clear()
    fib_sequence = generate_fibonacci(n)
    max_fib = max(fib_sequence)
    radius = max_fib
    circle = plt.Circle((0, 0), radius/8, edgecolor='grey', facecolor='none')
    ax.add_patch(circle)
    circle = plt.Circle((0, 0), math.log(max(1, radius)),
    edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    circle = plt.Circle((0, 0), radius, edgecolor='black', facecolor='none')
    ax.add_patch(circle)

    # Calculate positions and plot numbers around the circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_coords = [radius * math.cos(angle) for angle in angles]
    y_coords = [radius * math.sin(angle) for angle in angles]
    for j in [0,1]:
        for i, (x, y, fib) in enumerate(zip(x_coords, y_coords, fib_sequence)):
            if i % 2:
                ax.text((j*i*11) + x, (j*i*11) + y, f'{radius-fib}', color='black')
                ax.plot([(j*i*10), (j*i*10) - x/8], [0, y/8], 'green')
            else:
                ax.text((j*i*11) + x, (j*i*11) - y, f'{radius-fib}', color='grey')
                ax.plot([(j*i*10), (j*i*10) - x/8], [0, y/8], 'blue')
            ax.plot([(j*i*11), (j*i*11) - x/8], [0, -y/8], 'black')

    max_extent = max(max(abs(x) for x in x_coords), max(abs(y) for y in y_coords))
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.canvas.draw_idle()  
fig, ax = plt.subplots()  
plt.subplots_adjust(bottom=0.25)  
initial_n = 5
draw_circle_with_fib(initial_n)
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03]) 
slider = widgets.Slider(ax_slider, 'Num Fibonacci', 1, 35,
valinit=initial_n, valfmt='%0.0f')
slider.on_changed(draw_circle_with_fib)
plt.show()