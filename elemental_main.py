import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import periodictable

# Constants
sigma_SB = 5.67e-8  # Stefan-Boltzmann constant as sigma
radius = 0.001  # meters
num_points = 1000  # Increase for finer resolution in the spatial domain
load = 1.0  # arbitrary units
dt_initial = 1.0  # Initial dt value

# Get all elements
elements = [element for element in periodictable.elements if element.symbol]
element_symbols = [element.symbol for element in elements]
atomic_weights = [element.mass for element in elements]

# Initial dx set to the atomic weight of the first element
dx_initial = atomic_weights[0]

# Grid settings
grid_size = 1000
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x, y)

# Gaussian manifold definition
def gaussian_manifold(X, Y, x0, y0, sigma):
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# Calculate the derivatives
def calculate_derivatives(Z):
    Zx, Zy = np.gradient(Z, axis=0), np.gradient(Z, axis=1)
    Zxx, Zyy = np.gradient(Zx, axis=0), np.gradient(Zy, axis=1)
    return Zx, Zy, Zxx + Zyy

# Simulate the evolution
def simulate_evolution(dx, dt, timesteps=50):
    sigma = sigma_SB + dx  # Adjusting sigma based on dx for variability
    Z = gaussian_manifold(X, Y, 0, 0, sigma)
    Z_evolution = [Z]
    
    for t in range(timesteps):
        _, _, Laplacian_Z = calculate_derivatives(Z)
        Z = Z + dt * Laplacian_Z  # Simplified update step based on Laplacian
        Z_evolution.append(Z)
    
    return Z_evolution

def update(element):
    element_info = f"Element: {element.name}\nSymbol: {element.symbol}\nAtomic Number: {element.number}\nAtomic Mass: {element.mass}"
    text_box.set_text(element_info)
    qdot_L.cla()
    qdot_R.cla()
    qdot.cla()
    ax_jacobian.cla()
    ax5.cla()
    ax6.cla()
    ax7.cla()
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax31.cla()
    flow.cla()
    update.cached_x = None
    update.Z_evolution = None
    update.mid_index = None
    integer_part = int(element.mass)  # Get the integer part of the mass
    decimal_part = int((element.mass - integer_part) * 100000)  # Get the decimal part as an integer
    decimal_segments = [int(x) for x in str(decimal_part)]
    dx = integer_part
    dt = dx * np.pi
    if not hasattr(update, "cached_x") or update.cached_x != dx:
        update.cached_x = dx
        update.Z_evolution = simulate_evolution(dx, dt, timestep)
        update.mid_index = min(timestep - int(np.abs(np.log(timestep))), len(update.Z_evolution) - 1)

    # Define a function to determine color based on index
    def get_color(index):
        colors = ['black', 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'grey', 'pink', 'purple']
        return colors[index % len(colors)]

    i_index = 0
    while i_index < len(decimal_segments):
        i = decimal_segments[i_index]
        print(f"Segment value: {i}")
        x_index = 0
        while x_index < i:
            y_index = 0
            while y_index < i:
                x_pos = (x_index * np.cos(x_index**i))
                y_pos = (y_index * np.sin((i*np.pi/np.e)))
                z_pos = x_index**y_index*i
                flow.scatter(x_pos, y_pos, z_pos, 'o', color=get_color(np.abs(x_index*y_index*i)))
                flow.set_title(element.name)
                y_index += 1
            
            x_index += 1
        
        i_index += 1



    Z_evolution = simulate_evolution(dx, dt, timestep)
    titles = ["Initial Manifold", "Mid Evolution", "Final Evolved Manifold"]
    for ax, data, title in zip([ax5, ax6, ax7],
                               [update.Z_evolution[0], update.Z_evolution[update.mid_index], update.Z_evolution[-1]],
                               titles):
        ax.cla()  # Clear only the necessary plot
        logged_data = np.log(np.abs(data) + 1e-10)
        ax.plot_surface(X, Y, logged_data, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Value')

    # Single-valued Jacobian plot
    Zx, Zy, Laplacian_Z = calculate_derivatives(Z_evolution[-1])
    Jacobian = np.abs(Zx + Zy)
    ax_jacobian.cla()
    ax_jacobian.plot_surface(X, Y, Jacobian, cmap='inferno')
    ax_jacobian.set_title("Single-valued Jacobian")
    ax_jacobian.set_xlabel('X')
    ax_jacobian.set_ylabel('Y')
    ax_jacobian.set_zlabel('Jacobian Value')
    
    update_stress_distribution(dx, dt)
    # Efficiently update display, minimizing figure redraws
    fig.canvas.draw_idle()
    fig_buttons.canvas.draw_idle()

# Function to calculate and plot stress distribution
def update_stress_distribution(dx, dt, num_periods=11, period_length=2*np.pi):
    # Generate mesh points
    ax2.cla()
    x_stress = np.linspace(-radius, radius, num_points)
    y_stress = np.linspace(-radius, radius, num_points)
    X_stress, Y_stress = np.meshgrid(x_stress, y_stress)
    R = np.sqrt(X_stress**2 + Y_stress**2)
    # Periodic variation
    periods = np.linspace(0, period_length, num_periods)
    # Define the stress distribution with an additional complex component (e.g., 2i)
    stress = load * np.exp((dx * R**2) / (dt * (radius**2))) * (np.cos(dx * R) + 2j * np.sin(dt * np.log(R + 1e-11)))

    # Apply Fourier transform
    stress_fft = np.fft.fftshift(np.fft.fft2(stress))
    ln_t= np.log(np.abs(stress_fft) + 0.0001)*(8*np.pi/np.e)
    # Frequency domain visualization
    frequencies = np.fft.fftshift(np.fft.fftfreq(num_points, d=(x_stress[1] - x_stress[0])))
    FX, FY = np.meshgrid(frequencies, frequencies)    
    ax2.imshow(ln_t, extent=(frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]), origin='lower', cmap='viridis')
    for x in stress_fft:
        ax2.plot(x/frequencies, x/frequencies+12, 'b-')
    ax2.set_title('Magnitude of Stress Distribution in Frequency Domain')
    ax2.set_xlabel('Frequency x')
    ax2.set_ylabel('Frequency y')

    # Line plot correction
    for i, freq in enumerate(frequencies):
        ax31.plot(FX[i, :]*FY[i,:], (np.abs(stress_fft[i, :]) * (i*freq)/np.pi), 'b-')

    ax31.set_title('Super String Part of Stress Distribution in Spatial Domain')
    ax31.set_xlabel('x (gammaa)')
    ax31.set_ylabel('y (zeta)')
    ax31.set_zlabel('String')

    # Plot spatial domain in 3D (real part)
    ax1.plot_surface(X_stress, Y_stress, np.real(np.log(1e-11-stress)), cmap='viridis')
    ax1.set_title('Real Part of Stress Distribution in Spatial Domain')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('Stress')
    
    # Plot imaginary part of spatial domain in 3D
    ax3.plot_surface(X_stress, Y_stress, np.imag(np.log(1e-11-stress)), cmap='viridis')
    ax3.set_title('Imaginary Part of Stress Distribution in Spatial Domain')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_zlabel('Imaginary Stress')

# Plotting setup for the main figure
fig = plt.figure(figsize=(18, 18))
flow = fig.add_subplot(4,4, 1, projection='3d')
flow.view_init(0, -90, 0)
ax31 = fig.add_subplot(4, 4, 7, projection='3d')
ax2 = fig.add_subplot(4, 4, 4)
ax1 = fig.add_subplot(4, 4, 5, projection='3d')
qdot_L = fig.add_subplot(4, 4, 10, projection='3d')
qdot_R = fig.add_subplot(4, 4, 11, projection='3d')
qdot = fig.add_subplot(4, 4, 8, projection='3d')
ax_jacobian = fig.add_subplot(4, 4, 12, projection='3d')
ax3 = fig.add_subplot(4, 4, 13, projection='3d')
ax5 = fig.add_subplot(4, 4, 14, projection='3d')
ax6 = fig.add_subplot(4, 4, 15, projection='3d')
ax7 = fig.add_subplot(4, 4, 16, projection='3d')
# Initial plot
timestep = 50
Z_evolution = simulate_evolution(dx_initial, dt_initial, timestep)

# Calculate derivatives for the last timestep of evolution
Zx, Zy, Laplacian_Z = calculate_derivatives(Z_evolution[-1])

# Calculate the Jacobian (considering it as the magnitude of the gradient components)
Jacobian = np.abs(Zx + Zy)

# Plot the derivative with respect to X on qdot_L
qdot_L.plot_surface(X, Y, Zx, cmap='inferno')
qdot_L.set_title("Derivative with respect to X")
qdot_L.set_xlabel('X')
qdot_L.set_ylabel('Y')
qdot_L.set_zlabel('Zx Value')

# Plot the derivative with respect to Y on qdot_R
qdot_R.plot_surface(X, Y, Zy, cmap='inferno')
qdot_R.set_title("Derivative with respect to Y")
qdot_R.set_xlabel('X')
qdot_R.set_ylabel('Y')
qdot_R.set_zlabel('Zy Value')

# Plot the Laplacian on qdot
qdot.plot_surface(X, Y, Laplacian_Z, cmap='inferno')
qdot.set_title("Laplacian")
qdot.set_xlabel('X')
qdot.set_ylabel('Y')
qdot.set_zlabel('Laplacian Value')

# Plot the Jacobian on ax_jacobian
ax_jacobian.plot_surface(X, Y, Jacobian, cmap='inferno')
ax_jacobian.set_title("Single-valued Jacobian")
ax_jacobian.set_xlabel('X')
ax_jacobian.set_ylabel('Y')
ax_jacobian.set_zlabel('Jacobian Value')

# Call the update stress distribution function with initial dx and dt
update_stress_distribution(dx_initial, dt_initial)

# Button setup in a new figure
fig_buttons = plt.figure(figsize=(18, 12))
button_axes = []
buttons = []

# Create a mapping of periodic table layout (row, column)
periodic_table_layout = {
    'H': (0, 0), 'He': (0, 17),
    'Li': (1, 0), 'Be': (1, 1), 
    'B': (1, 12), 'C': (1, 13), 'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17),
    'Na': (2, 0), 'Mg': (2, 1), 
    'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17),
    'K': (3, 0), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7), 'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11), 
    'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14), 'Se': (3, 15), 'Br': (3, 16), 'Kr': (3, 17),
    'Rb': (4, 0), 'Sr': (4, 1), 'Y': (4, 2), 'Zr': (4, 3), 'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6), 'Ru': (4, 7), 'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11),
    'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14), 'Te': (4, 15), 'I': (4, 16), 'Xe': (4, 17),
    'Cs': (5, 0), 'Ba': (5, 1), 
    'Hf': (5, 3), 'Ta': (5, 4), 'W': (5, 5), 'Re': (5, 6), 'Os': (5, 7), 'Ir': (5, 8), 'Pt': (5, 9), 'Au': (5, 10), 'Hg': (5, 11),'Tl': (5, 12), 'Pb': (5, 13), 'Bi': (5, 14), 'Po': (5, 15), 'At': (5, 16), 'Rn': (5, 17),
    'Fr': (6, 0), 'Ra': (6, 1),    
    'Rf': (6, 3), 'Db': (6, 4), 'Sg': (6, 5), 'Bh': (6, 6), 'Hs': (6, 7), 'Mt': (6, 8), 'Ds': (6, 9), 'Rg': (6, 10), 'Cn': (6, 11), 'Nh': (6, 12), 'Fl': (6, 13), 'Mc': (6, 14), 'Lv': (6, 15), 'Ts': (6, 16), 'Og': (6, 17),
    'La': (7, 2), 'Ce': (7, 3), 'Pr': (7, 4), 'Nd': (7, 5), 'Pm': (7, 6), 'Sm': (7, 7), 'Eu': (7, 8), 'Gd': (7, 9), 'Tb': (7, 10), 'Dy': (7, 11), 'Ho': (7, 12), 'Er': (7, 13), 'Tm': (7, 14), 'Yb': (7, 15), 'Lu': (7, 16),
    'Ac': (8, 2), 'Th': (8, 3), 'Pa': (8, 4), 'U': (8, 5), 'Np': (8, 6), 'Pu': (8, 7), 'Am': (8, 8), 'Cm': (8, 9), 'Bk': (8, 10), 'Cf': (8, 11),  'Es': (8, 12), 'Fm': (8, 13), 'Md': (8, 14), 'No': (8, 15), 'Lr': (8, 16),
}

radioactive_elements = {
    'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 
    'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 
    'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}

# Add a text box to display the selected element information
info_ax = fig_buttons.add_axes([0.02, 0.02, 0.3, 0.1])
info_ax.axis('off')
text_box = info_ax.text(1, 9, '', transform=info_ax.transAxes, ha='center', va='center', fontsize=12)

for element in elements:
    symbol = element.symbol
    if symbol in periodic_table_layout:
        row, col = periodic_table_layout[symbol]
        ax = fig_buttons.add_axes([0.02 + col * 0.045, 0.85 - row * 0.045, 0.04, 0.04])
        color = 'yellow' if symbol in radioactive_elements else 'lightgrey'
        button = Button(ax, element.symbol, color=color, hovercolor='blue')
        button.on_clicked(lambda event, el=element: update(el))
        buttons.append(button)

plt.show()