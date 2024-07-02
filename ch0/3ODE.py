import math
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from mpl_toolkits.mplot3d import Axes3D
#import mpl_interactions.ipyplot as iplt
def generate_primes(n):
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    p = 2
    while p * p <= n:
        if primes[p]:
            for i in range(p * p, n + 1, p):
                primes[i] = False
        p += 1
    prime_numbers = [x for x in range(2, n + 1) if primes[x]]
    return prime_numbers
##############
# vector v1+ iv2 = 0
# V = [v1 v2] = [-it t] = [1 i] 
# therefore we deduce y info only over reals
#     x = (pi*sin(x_0)) + (pi/(e(cos(x_1)))
#     y = (pi/(e*cos(y_0)))) - (phi*sin(y_1))
# as a function onto the complex in any t
#    x_0 -> (x_1 /\ y_1) -> y_2
# where;
#   theta -> (phi -> phi) -> theta
def plotdy(theta, phi):
	y = (r/(e(np.cos(theta)))) - (R*np.sin(phi))
	x = theta 
	z = (np.cos(theta) + e(np.sin(theta)))
	return x, y, z
def plotdx(theta, phi):
	y = theta
	x = (R*np.sin(phi)) + (r/(e(np.cos(theta)))) 
	z = (np.cos(theta) + e(np.sin(theta)))
	return x, y, z
	
# function to show superposition 0 t from two complex observation points [1, -1i] with its axions
# Define the Fibonacci sequence
fibseq = [1, 1]
# Define iterations -------------------------------------------------------------------------
n = 5

# Torus parameters
R = (n + np.sqrt(5)) / 2**n # Major radius
r = np.pi  # Minor observable radius
e = np.exp #Sub minor obersables

# where r/c is a Real fn on (2)^{0.5}

# Draw the objects
fig, ax = plt.subplots(1, 1, figsize=(2*n, 2*n))
# Create the inset axes as a 3D axes object
axnotT = fig.add_axes([0.1, 0.3, 0.2, 0.2], projection='3d', proj_type='ortho')  # Adjust position and size

# Create the right subplot as a 3D axes object
axT = fig.add_subplot(122, projection='3d', proj_type='ortho')
axnotT.axis('off')
axT.axis('off')
axnotT.set_facecolor('none') # Set transparency for left subplot
axT.set_facecolor('none')  # Set transparency for right subplot
axnotT.view_init(elev=n, azim=-n)
axT.view_init(elev=n, azim=n)

# Number of additional objects to add
primes = generate_primes(n)
# Create torus points
phi, theta = np.mgrid[0:2*r:100j, 0:2*r:100j]

x,y,z = plotdx(-theta, -phi)
axnotT.plot_surface(x, y, z, color='y', alpha=1)
#controls = iplt.plot(n, x, r=r, beta=(1, 10, 100), label="not t")

x,y,z = plotdy(theta, phi)
axT.plot_surface(x, y, z, color='b', alpha=1)
#iplt.plot(theta, x, controls=controls, label="t")

# Planck Number line
axT.plot([-n, n], [0, 0], 'k-', linewidth=1)
# Complexity number line
axT.plot([ n*math.e, math.log(n*math.pi)], [n*math.e, math.log(n*math.pi)], 'k-', linewidth=1)
# Real number line
axnotT.plot([0, -n], [0 , n], 'k-', linewidth=1)
# U Real CONSTRUCTOR 
realConstructor = plt.Circle((0, 0), math.pi, color='black', fill=False)
ax.add_artist(realConstructor)
# U Complex CONSTRUCTOR
complexConstructor = plt.Circle((0, 0), math.e, color='red', fill=False)
ax.add_artist(complexConstructor)
# U Observer Complex CONSTRUCTOR
observationConstructor = plt.Circle((0, 0), math.log(math.pi*math.e), color='green', fill=True, alpha=0.3)
ax.add_artist(observationConstructor)
# U Observer Real CONSTRUCTOR
observationConstructor = plt.Circle((0, 0), math.pi**math.log(math.pi), color='blue', fill=True, alpha=0.1)
ax.add_artist(observationConstructor)
# U+1 = Real Infomation
realInfomation = plt.Circle((0, 0),  (math.pi)**math.e, color='black', fill=False)
ax.add_artist(realInfomation)
# U+1 = Complex Infomation 
complexInfomation = plt.Circle((0, 0), (math.e)**math.pi, color='black', fill=False)
ax.add_artist(complexInfomation)
# U+1 = Observer Infomation [2 units of info + relitive complexity]
observationConstructor = plt.Circle((0, 0), math.log((math.pi)**math.e*(math.e)**math.pi), color='black', fill=True, alpha=0.1)
ax.add_artist(observationConstructor)
# U-1 = INTERNAL Real subCONSTRUCTOR
internalRealConstructor = plt.Circle((0, 0), (math.pi/math.e), color='pink', fill=False)
ax.add_artist(internalRealConstructor)
# U-1 = INTERNAL Complex subCONSTRUCTOR
internalComplexConstructor = plt.Circle((0, 0), (math.e/math.pi), color='pink', fill=False)
ax.add_artist(internalComplexConstructor)
# U-1 = Internal Observer subCONSTRUCTOR = ZERO - no process
internalObservationConstructor = plt.Circle((0, 0), math.log((math.pi/math.e)*(math.e/math.pi)), color='pink', fill=True, alpha=1)
ax.add_artist(internalObservationConstructor)

# Set plot parameters
ax.set_xlabel('X-axis')
axnotT.set_ylabel('Y-axis')
axT.set_zlabel('Z-axis')
ax.set_title('t object with observable Loops')
x_min, x_max = axnotT.get_xlim()
y_min, y_max = axT.get_ylim()
new_x_min = x_min - 0.1 * (x_max - x_min)
new_x_max = x_max + 0.1 * (x_max - x_min)
new_y_min = y_min - 0.1 * (y_max - y_min)
new_y_max = y_max + 0.1 * (y_max - y_min)

ax.set_xlim(new_x_min, new_x_max)
ax.set_ylim(new_y_min, new_y_max)
for i in range(2, n + 2):
    fibseq.append(fibseq[i - 1] + fibseq[i - 2])

# Calculate the Real Infomation Storage from notT
for i in range(0, n+1):
    ax.annotate(f'{i}', xy=(-i, i), xytext=(-i-1, i+1),
                arrowprops=dict(arrowstyle='->'))

# Calculate the Infomation Storage in Orical
for i in range(1, n):
    for j in range(1, fibseq[i-1]):
        x =  j * (math.pi/math.e)
        y = i * (math.pi/math.e)
        circle = plt.Circle((x, y), (1/2**n), color='black', fill=False)
        ax.add_artist(circle)

# Add efficency Complex Storage
for j in range(0, len(primes)):
        circle = plt.Circle((primes[j], primes[j]), 2/n, color='green', fill=True)
        ax.add_artist(circle)

# Calculate the Plank storage for Real Storage 
for i in range(1, n):
    j = math.log(n * math.e) / (i * math.pi)
    x = math.log((j * (math.pi / math.e)))
    y = i * (math.pi / math.e)
    # Draw lines of Complex Unit Values to the projected observed points in U
    ax.plot([x, x], [0, -x], 'r--', linewidth=0.5)  # Vertical line from infomation observation Unit
    ax.plot([x, y], [0, -x], 'b--', linewidth=0.5)  # Diazgonal line from observation to infomation Unit
    ax.plot([y, y], [0, -x], 'g--', linewidth=0.5)  # Horizontal line from observation point to complex relative infomation Unit
    if math.log2(i) % 1 == 0:
        circle = plt.Circle((0, 0), i, edgecolor='r', facecolor='none', linewidth=1) # Shells of Infomation
        ax.add_patch(circle)

#_ = plt.legend()
plt.show()
