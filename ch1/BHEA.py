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

# Function to generate the Fibonacci sequence using Euler's formula
# This function uses the golden ratio to compute the Fibonacci sequence
# The sequence is used to determine the phases for the quantum circuit
def generate_fibonacci_euler(n):
    fib_seq = []
    sqrt_5 = np.sqrt(5)
    phi = (1 + sqrt_5) / 2
    psi = (1 - sqrt_5) / 2
    for i in range(n):
        fib_number = (phi**i - psi**i) / sqrt_5
        fib_seq.append(fib_number)
    return fib_seq

# Class to simulate a primitive result object
# This class stores the data and metadata of the simulation results
class PrimitiveResult:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

# Class to simulate a publication result object
# This class stores the data and metadata of the publication results
class PubResult:
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata

# Class to store data bins for the simulation
# This class holds the expected values, standard deviations, and ensemble standard errors
class DataBin:
    def __init__(self, evs, stds, ensemble_standard_error):
        self.evs = evs
        self.stds = stds
        self.ensemble_standard_error = ensemble_standard_error

# Initialize the Qiskit runtime service
# This service is used to interface with IBM Quantum's backend
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='0bdcd9d680f9009e85d94da9d68ca8e3c59a3443e78f99f5a3cca86331dd22045aec92623d0e255a501a834e23b8798fc4b96c7c686714c6d991bfbf496ec18d'  # Insert your real token here
)

# Select the backend for simulation
backend = service.backend('ibm_brisbane')
target = backend.target

# Input for number of qubits and depth
# These inputs are used to configure the quantum circuit
num_qubit = int(input("Enter the number of qubits: "))
depth = int(input("Enter the depth of the circuit: "))
theta = Parameter('θ')

# Generate Fibonacci sequence for the phases using Euler's formula
# The sequence is used to set the phases in the quantum circuit
fib_phases = generate_fibonacci_euler(num_qubit * depth)

# Prepare the quantum circuit
# This section sets up the quantum circuit with the specified number of qubits and depth
qc = QuantumCircuit(num_qubit)

# Populate the circuit with gates
# Apply X gates to even qubits, followed by CZ and U gates in layers
for q in range(0, num_qubit, 2):
    qc.x(q)
for d in range(depth):
    for i in range(0, num_qubit, 2):
        if i + 1 < num_qubit:
            qc.cz(i, i + 1)
        qc.u(theta, 0, -np.pi, i)
        if i + 1 < num_qubit:
            qc.u(theta, 0, -np.pi, i + 1)
    for q in range(num_qubit):
        qc.p(fib_phases[d * num_qubit + q], q)

# Reverse the process to complete the circuit
# This ensures the circuit goes back to the initial state
for d in range(depth - 1, -1, -1):
    for q in range(num_qubit - 1, -1, -1):
        qc.p(-fib_phases[d * num_qubit + q], q)
    for i in range(num_qubit - 1, -1, -2):
        if i - 1 >= 0:
            qc.u(-theta, 0, np.pi, i)
            qc.u(-theta, 0, np.pi, i - 1)
        else:
            qc.u(-theta, 0, np.pi, i)
    for i in range(num_qubit - 1, -1, -2):
        if i - 1 >= 0:
            qc.cz(i - 1, i)
for q in range(0, num_qubit, 2):
    qc.x(q)
print(qc)

# Define the observable and transpile the circuit
# The observable is a Pauli-Z operator and the circuit is optimized for the target backend
obs = SparsePauliOp('I' + 'Z' + 'I' * (num_qubit - 2))
pm = generate_preset_pass_manager(target=target, optimization_level=3)
t_qc = pm.run(qc)
t_obs = obs.apply_layout(t_qc.layout)

# Simulation parameters
# Set random seed and generate random parameter values for the simulation
np.random.seed(0)
parameter_values = np.random.uniform(-np.pi/2, np.pi/2, num_qubit)  # Correcting number of values

# Initialize the estimator for running the simulation
estimator = EstimatorV2(backend=backend)

# Prepare inputs and run estimation
# Execute the simulation and retrieve the result object
inputs = (t_qc, t_obs, parameter_values)
job = estimator.run([inputs])  # Ensure correct structure
result = job.result()  # Get the result object
print("Result object:", result)
evs_value = result[0].data.evs
print("EVS:", evs_value)

# Calculate the square root of the absolute EVS values
# This transformation is applied for visualization purposes
sqrt_evs = [np.sqrt(abs(evs)) for evs in evs_value]

# Calculate the mode of the EVS values
# The mode is used as a representative value in the visualization
mode_evs_result = stats.mode(evs_value)
mode_evs = mode_evs_result.mode[0] if isinstance(mode_evs_result.mode, np.ndarray) else mode_evs_result.mode
mode_evs_sqrt = np.sqrt(abs(mode_evs))

# Plot the EVS values and their square roots in 3D with color hue for the position vector
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Define the x, y, z coordinates for the plot
x = range(len(evs_value))  # Using index as x-axis
y = evs_value
z = sqrt_evs
colors = plt.cm.viridis(np.linspace(0, 1, len(evs_value)))  # Color hue

# Scatter plot of the EVS values
ax.scatter(x, y, z, c=colors, marker='o')

# Connecting the points using the cos^2(theta) function with higher resolution
# This function is used to visualize the connection between points
theta_values = np.linspace(0, 2 * np.pi, 1000)  # Increase the number of points for higher resolution
cos_squared = np.cos(theta_values)**2  # real part
sin_squared = np.sin(theta_values)**2  # complex part

# Interpolate x and y values for higher resolution
x_high_res = np.linspace(min(x), max(x), 1000)
y_high_res = np.interp(x_high_res, x, y)

# Define and plot the different routes
Hgate = [[x_high_res, y_high_res], [x_high_res, y_high_res]]
ax.plot(Hgate[0][0], Hgate[0][1], cos_squared, color='black', linestyle='-')  # real route
ax.plot(Hgate[1][0], Hgate[1][1], sin_squared, color='red', linestyle='--')  # complex symmetry route
ax.plot(x_high_res, y_high_res, color='blue', linestyle='solid')  # effective route

# Creating planes connecting the vectors
# Planes are added to connect adjacent points for better visualization
for i in range(len(x)-1):
    xx, yy = np.meshgrid([x[i], x[i+1]], [y[i], y[i+1]])
    zz = np.array([[z[i], z[i]], [z[i+1], z[i+1]]])
    ax.plot_surface(xx, yy, zz, alpha=0.2, facecolors=plt.cm.viridis(np.linspace(0, 1, 2).reshape(-1, 1)))

# Adding a plane from the trigonometric identity to the nodes with a hue color for gradient
# This plane visualizes the relationship between the trigonometric function and the EVS values
X, Y = np.meshgrid(x, np.linspace(min(z), max(z), 100))
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
ax.set_title('3D Plot of EVS Values and Their Square Roots with Color Hue and cos^2(theta)')

plt.show()

# Create a table of EVS values and their corresponding square roots
# The table is displayed as a pandas DataFrame
data = {'EVS Value': evs_value, 'Square Root': sqrt_evs}
df = pd.DataFrame(data)
print(df)

# Simulate the result object
# This section simulates the result object for further analysis
result = PrimitiveResult(
    data=[
        PubResult(
            data=DataBin(
                evs= result[0].data.evs, 
                stds= result[0].data.stds, 
                ensemble_standard_error= result[0].data.ensemble_standard_error
            ), 
            metadata={
                'shots': 2024, 
                'target_precision': 0.96, 
                'circuit_metadata': {}, 
                'resilience': {}, 
                'num_randomizations': 32
            }
        )
    ],
    metadata={
        'dynamical_decoupling': {
            'enable': False, 
            'sequence_type': 'XX', 
            'extra_slack_distribution': 'middle', 
            'scheduling_method': 'alap'
        }, 
        'twirling': {
            'enable_gates': False, 
            'enable_measure': True, 
            'num_randomizations': 'auto', 
            'shots_per_randomization': 'auto', 
            'interleave_randomizations': True, 
            'strategy': 'active-accum'
        }, 
        'resilience': {
            'measure_mitigation': True, 
            'zne_mitigation': False, 
            'pec_mitigation': False
        }, 
        'version': 2
    }
)

# Extract data values from the result object
# The data values are used to create a graph representation
pub_result = result.data[0]
data_value = DataBin(
    evs=pub_result.data.evs,
    stds=pub_result.data.stds,
    ensemble_standard_error=pub_result.data.ensemble_standard_error
)

# Create a graph from the result data
# The graph visualizes the relationships between different data points
G = nx.Graph()

# Adding nodes to the graph
G.add_node(1, label='evs', value=data_value.evs)
G.add_node(2, label='stds', value=data_value.stds)
G.add_node(3, label='ensemble_standard_error', value=data_value.ensemble_standard_error)

# Adding edges to the graph
edges = [(1, 2), (1, 3)]
G.add_edges_from(edges)

# Plotting the graph
# The graph is visualized using NetworkX and Matplotlib
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
