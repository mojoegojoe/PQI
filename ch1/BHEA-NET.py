# 0bdcd9d680f9009e85d94da9d68ca8e3c59a3443e78f99f5a3cca86331dd22045aec92623d0e255a501a834e23b8798fc4b96c7c686714c6d991bfbf496ec18d
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Options
from qiskit.circuit import Parameter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# Initialize the Qiskit runtime service
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='0bdcd9d680f9009e85d94da9d68ca8e3c59a3443e78f99f5a3cca86331dd22045aec92623d0e255a501a834e23b8798fc4b96c7c686714c6d991bfbf496ec18d'  # Insert your real token here
)

# Fetch the result of the job
job = service.job('cs2009hkfpw00080egv0')
result = job.result()
print("Result object:", result)
evs_value = result[0].data.evs
print("EVS:", evs_value)

def generate_ka_plots(evs_value, d, n, beta, m):
    a = np.random.rand(d)
    bp = np.random.rand(n, d)
    cq = np.random.rand(n)
    
    def psi(x, q):
        # Sum over the first axis to reduce the shape from (2, 100, 100) to (100, 100)
        return np.tensordot(bp[q], x, axes=(0, 0)) + (q * a[:, None, None]).sum(axis=0)
    
    def g(x):
        sum_term = np.zeros(x[0].shape)  # Initialize sum_term with the correct shape
        for q in range(n):
            sum_term += evs_value[q % len(evs_value)] * psi(x, q)
        return sum_term / (n + beta * m ** d)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
    
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Interpolate evs_value to create a 2D grid of the same size as X1 and X2
    Z_original = np.interp(X1.flatten(), np.linspace(0, 1, len(evs_value)), evs_value).reshape(X1.shape)
    
    # Compute g over the meshgrid and ensure the result is reshaped to 2D
    G_input = np.array([X1, X2])
    Z_approx = g(G_input)
    
    # Debugging statements to check the shape of the results
    print("X1 shape:", X1.shape)
    print("X2 shape:", X2.shape)
    print("G_input shape:", G_input.shape)
    print("Z_approx shape before reshaping:", Z_approx.shape)
    
    # Reshape Z_approx to match the shape of X1
    Z_approx = Z_approx.reshape(X1.shape)
    print("Z_approx shape after reshaping:", Z_approx.shape)
    
    # Original function plot
    axs[0, 0].contourf(X1, X2, Z_original, cmap='viridis')
    axs[0, 0].set_title('Original Function')
    
    # Approximated function plot
    axs[0, 1].contourf(X1, X2, Z_approx, cmap='viridis')
    axs[0, 1].set_title('Approximated Function')
    
    # Difference plot
    diff = np.abs(Z_original - Z_approx)
    axs[1, 0].contourf(X1, X2, diff, cmap='viridis')
    axs[1, 0].set_title('Difference')
    
    # Error plot
    error = np.mean(diff)
    axs[1, 1].text(0.5, 0.5, f'Mean Error: {error:.4f}', horizontalalignment='center', verticalalignment='center', fontsize=15)
    axs[1, 1].set_title('Error')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
# Calculate the square root of the absolute EVS values
sqrt_evs = [np.sqrt(abs(evs)) for evs in evs_value]

# Calculate the mode of the EVS values
mode_evs_result = stats.mode(evs_value)
mode_evs = mode_evs_result.mode[0] if isinstance(mode_evs_result.mode, np.ndarray) else mode_evs_result.mode
mode_evs_sqrt = np.sqrt(abs(mode_evs))

# Plot the EVS values and their square roots in 3D with color hue for the position vector
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Define the x, y, z coordinates for the plot
x = range(len(evs_value))
y = evs_value
z = sqrt_evs
colors = plt.cm.viridis(np.linspace(0, 1, len(evs_value)))

# Scatter plot of the EVS values
ax.scatter(x, y, z, c=colors, marker='o')

# Connecting the points using the cos^2(theta) function with higher resolution
theta_values = np.linspace(0, 2 * np.pi, 1000)
cos_squared = np.cos(theta_values)**2
sin_squared = np.sin(theta_values)**2

# Interpolate x and y values for higher resolution
x_high_res = np.linspace(min(x), max(x), 1000)
y_high_res = np.interp(x_high_res, x, y)

# Define and plot the different routes
Hgate = [[-x_high_res, y_high_res**2], [x_high_res**2, -y_high_res]]
ax.plot(Hgate[0][0], Hgate[0][1], cos_squared, color='black', linestyle='-')
ax.plot(Hgate[1][0], Hgate[1][1], sin_squared, color='red', linestyle='--')
ax.plot(Hgate[1][0], Hgate[1][1], cos_squared + sin_squared, color='red', linestyle='--')
ax.plot(x_high_res, y_high_res, color='blue', linestyle='solid')

# Creating planes connecting the vectors
for i in range(len(x) - 1):
    xx, yy = np.meshgrid([x[i], x[i + 1]], [y[i], y[i + 1]])
    zz = np.array([[z[i], z[i]], [z[i + 1], z[i + 1]]])
    ax.plot_surface(xx, yy, zz, alpha=0.2, facecolors=plt.cm.viridis(np.linspace(0, 1, 2).reshape(-1, 1)))

# Adding a plane from the trigonometric identity to the nodes with a hue color for gradient
X, Y = np.meshgrid(x, np.linspace(min(y), max(y), 100))
Z = np.cos(Y)**2

# Create a colormap for the plane
norm = plt.Normalize(np.min(Z), np.max(Z))
colors_plane = plt.cm.viridis(norm(Z))

# Plot the surface with the color gradient
ax.plot_surface(X, Y, Z, facecolors=colors_plane, alpha=0.5)

# Set labels and title for the plot
ax.set_xlabel('Index')
ax.set_ylabel('EVS Values')
ax.set_zlabel('Square Roots of EVS Values')
ax.set_title('Plot of EVS Values and Their Square Roots with Color for Weight of Connection to t [cos^2(theta)+sin^2(theta)]')

plt.show()

# Create a table of EVS values and their corresponding square roots
data = {'EVS Value': evs_value, 'Square Root': sqrt_evs}
df = pd.DataFrame(data)
print(df)

# Create a graph from the result data
G = nx.Graph()

# Adding nodes to the graph
G.add_node(1, label='evs', value=result[0].data.evs)
G.add_node(2, label='stds', value=result[0].data.stds)
G.add_node(3, label='ensemble_standard_error', value=result[0].data.ensemble_standard_error)

# Adding edges to the graph
edges = [(1, 2), (1, 3)]
G.add_edges_from(edges)

# Plotting the graph
pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, 'label')
values = nx.get_node_attributes(G, 'value')

nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(1, 2): 'evs-stds', (1, 3): 'evs-ese'})
for node, (x, y) in pos.items():
    value = values[node]
    if isinstance(value, np.ndarray):
        text = '\n'.join([f"{v:.2e}" for v in value])
    else:
        text = f"{value:.2e}"
    plt.text(x, y - 0.1, s=text, bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')

plt.title('Graph Representation of Quantum Result')
plt.show()

# Generate Kolmogorov-Arnold plots using the quantum computation output results
generate_ka_plots(evs_value, d=2, n=10, beta=0.5, m=5)
