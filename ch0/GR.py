import matplotlib.pyplot as plt
import numpy as np

# Define the grid
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)

# Define the mass and the curvature
mass = 1
Z = -mass / np.sqrt(X**2 + Y**2 + 1)

# Plotting the curvature of spacetime
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis_r')

# Labels and title
ax.set_title('Curvature of Spacetime Around a Massive Object')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Spacetime Curvature')

plt.show()
