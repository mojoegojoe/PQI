import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Parameters
radius = 0.001  # meters
num_points = 100  # Increase for finer resolution in the spatial domain
load = 1.0  # arbitrary units
dx_initial = 1.0  # Initial dx value
dt_initial = 1.0  # Initial dt value

# Function to calculate and plot stress distribution
def update_plot(dx, dt):
    # Generate mesh points
    x = np.linspace(-radius, radius, num_points)
    y = np.linspace(-radius, radius, num_points)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Define the stress distribution
    stress = load * np.exp((dx * R**2) / (dt * (radius**2))) * np.cos(dx * R) * np.sin(dt * np.log(R + 1e-11))

    # Apply Fourier transform
    stress_fft = np.fft.fftshift(np.fft.fft2(stress))

    # Frequency domain visualization
    frequencies = np.fft.fftshift(np.fft.fftfreq(num_points, d=(x[1] - x[0])))
    FX, FY = np.meshgrid(frequencies, frequencies)

    # Clear previous plots
    ax1.cla()
    ax2.cla()

    # Plot spatial domain in 3D
    ax1.plot_surface(X, Y, stress, cmap='viridis')
    ax1.set_title('Stress Distribution in Spatial Domain')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('Stress')

    # Plot frequency domain
    ax2.imshow(np.log(np.abs(stress_fft) + 1), extent=(frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]), origin='lower', cmap='viridis')
    ax2.set_title('Stress Distribution in Frequency Domain')
    ax2.set_xlabel('Frequency x')
    ax2.set_ylabel('Frequency y')

    fig.canvas.draw_idle()

# Plotting setup
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)
plt.subplots_adjust(left=0.1, bottom=0.35)

# Initial plot
update_plot(dx_initial, dt_initial)

# Slider setup
axcolor = 'lightgoldenrodyellow'
ax_dx = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_dt = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)

s_dx = Slider(ax_dx, 'dx', 0.0, 3.0, valinit=dx_initial)
s_dt = Slider(ax_dt, 'dt', 0.0, 5.0, valinit=dt_initial)

# Update plot based on slider value changes
s_dx.on_changed(lambda val: update_plot(s_dx.val, s_dt.val))
s_dt.on_changed(lambda val: update_plot(s_dx.val, s_dt.val))

plt.show()