import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import unittest

def fibonacci(n):
    """
    Generate a list of the first n Fibonacci numbers.
    
    Parameters:
    n (int): The number of Fibonacci numbers to generate.
    
    Returns:
    list: A list containing the first n Fibonacci numbers.
    """
    if n == 0:
        return []
    elif n == 1:
        return [1]
    fib_sequence = [1, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

def is_prime(num):
    """
    Check if a number is prime.
    
    Parameters:
    num (int): The number to check for primality.
    
    Returns:
    bool: True if the number is prime, False otherwise.
    """
    if num < 2:
        return False
    for i in range(2, int(np.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def plot_fibonacci_structures(num_fibonacci, fib_sequence, ax, title_suffix=''):
    """
    Plot Fibonacci numbers and their associated curves in a 3D space.
    
    Parameters:
    num_fibonacci (int): The number of Fibonacci numbers to plot.
    fib_sequence (list): The list of Fibonacci numbers.
    ax (Axes3D): The 3D axis to plot on.
    title_suffix (str): Suffix for the plot title.
    """
    x_values = np.arange(num_fibonacci)
    
    # Limit the range to prevent overflow
    limited_x_values = x_values[x_values < 666]  # Adjust the limit as needed
    y_values_cos_squared = np.cos((limited_x_values * np.pi / np.e))**2
    y_values_sin_squared = np.sin((limited_x_values * np.pi / np.e))**2
    y_values_tanh = np.tanh((limited_x_values * np.pi / np.e))
    
    # Apply scaling to hyperbolic functions to prevent overflow
    z_values_cosh = np.cosh((limited_x_values * np.pi / np.e)) / 1e5
    z_values_sinh = np.sinh((limited_x_values * np.pi / np.e)) / 1e5
    z_values_tanh = np.tanh((limited_x_values * np.pi / np.e))
    
    ax.clear()
    ax.scatter(limited_x_values, y_values_cos_squared, z_values_cosh, c='orange', label='cos^2 Curve', s=num_fibonacci*10/3)
    ax.scatter(limited_x_values, y_values_sin_squared, z_values_sinh, c='blue', label='sin^2 Curve', s=num_fibonacci*5)
    ax.scatter(limited_x_values, y_values_tanh, z_values_tanh, c='green', label='tanh Curve', s=num_fibonacci*10)

    prime_coords = []
    
    for i, fib_number in enumerate(fib_sequence):
        if i < len(limited_x_values):
            if is_prime(fib_number):
                prime_coords.append((limited_x_values[i], y_values_cos_squared[i], z_values_cosh[i]))
                ax.scatter(limited_x_values[i], y_values_cos_squared[i], z_values_cosh[i], color='green', edgecolor='black', s=100, label='Prime Fibonacci' if i == 0 else "")
                ax.scatter(limited_x_values[i], y_values_sin_squared[i], z_values_sinh[i], color='green', edgecolor='black', s=100)
                ax.scatter(limited_x_values[i], y_values_tanh[i], z_values_tanh[i], color='green', edgecolor='black', s=100)
                ax.text(limited_x_values[i], y_values_cos_squared[i], z_values_cosh[i], f'Prime: {fib_number}', color='black')
            else:
                ax.scatter(limited_x_values[i], y_values_cos_squared[i], z_values_cosh[i], color='orange', marker='o', alpha=0.7)
                ax.scatter(limited_x_values[i], y_values_sin_squared[i], z_values_sinh[i], color='blue', marker='o', alpha=0.7)
                ax.scatter(limited_x_values[i], y_values_tanh[i], z_values_tanh[i], color='green', marker='o', alpha=0.7)

    for i in range(num_fibonacci):
        if i < len(limited_x_values):
            ax.plot([limited_x_values[i], limited_x_values[i]], [y_values_cos_squared[i], y_values_sin_squared[i]], [z_values_cosh[i], z_values_sinh[i]], 'k-', alpha=0.5)
            ax.plot([limited_x_values[i], limited_x_values[i]], [y_values_sin_squared[i], y_values_tanh[i]], [z_values_sinh[i], z_values_tanh[i]], 'k-', alpha=0.5)

    # Plot complex density manifold connecting prime Fibonacci points
    if len(prime_coords) > 1:
        prime_coords = np.array(prime_coords)
        ax.plot(prime_coords[:, 0], prime_coords[:, 1], prime_coords[:, 2], 'r-', linewidth=2, label='Prime Connection')

    ax.set_title(f'Fibonacci Numbers with Corresponding Curves {title_suffix}')
    ax.set_xlabel('Index of Fibonacci Number')
    ax.set_ylabel('Function Value (y)')
    ax.set_zlabel('Hyperbolic Function Value (z)')
    ax.grid(True)
    ax.legend(loc='upper right')

# Initial parameters for Fibonacci sequence
initial_num_fibonacci = 3  # Start with 3 Fibonacci numbers

# Create the initial Fibonacci sequences
fib_numbers = fibonacci(initial_num_fibonacci)
# Create the plot
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot for Fibonacci sequence up to initial_num_fibonacci
plot_fibonacci_structures(initial_num_fibonacci, fib_numbers, ax, f'(Fn({initial_num_fibonacci}))')

# Add a slider for interactive control of num_fibonacci
ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Num Fibonacci', 0, 43, valinit=initial_num_fibonacci, valstep=1)

def update(val):
    num_fibonacci = int(slider.val)
    fib_numbers = fibonacci(num_fibonacci)
    plot_fibonacci_structures(num_fibonacci, fib_numbers, ax, f'(Fn({num_fibonacci}))')
    
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.tight_layout()
plt.show()

# Unit tests to validate the functions
class TestFibonacciAndPrimes(unittest.TestCase):
    def test_fibonacci(self):
        self.assertEqual(fibonacci(0), [])
        self.assertEqual(fibonacci(1), [1])
        self.assertEqual(fibonacci(2), [1, 1])
        self.assertEqual(fibonacci(5), [1, 1, 2, 3, 5])
    
    def test_is_prime(self):
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(1))
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertFalse(is_prime(4))
        self.assertTrue(is_prime(5))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
