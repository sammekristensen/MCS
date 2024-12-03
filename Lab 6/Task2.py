import numpy as np
import matplotlib.pyplot as plt
import random

# Define the three points of the triangle
point1 = np.array([0, 0])
point2 = np.array([1, 0])
point3 = np.array([0.5, np.sqrt(3)/2])

# Function to generate a random point inside the triangle
def random_point_in_triangle(p1, p2, p3):
    s, t = sorted([random.random(), random.random()])
    return s * p1 + (t - s) * p2 + (1 - t) * p3

# Initial current position inside the triangle
current_position = random_point_in_triangle(point1, point2, point3)

# Define probabilities for each vertex
p1, p2, p3 = 0.1, 0.1, 0.8

# List to store the points after 100 steps
points = []

# Perform the iteration
for i in range(5000):  # Let's do 5000 iterations for a good plot
    # Randomly select one of the three vertices based on the probabilities
    vertex = random.choices([point1, point2, point3], weights=[p1, p2, p3], k=1)[0]
    
    # Move half the distance from the current position to the selected vertex
    current_position = (current_position + vertex) / 2
    
    # After the first 100 steps, start storing the points
    if i >= 100:
        points.append(current_position)

# Convert the list of points to a numpy array for plotting
points = np.array(points)

# Plot the points
plt.scatter(points[:, 0], points[:, 1], s=0.1)
plt.show()
