import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Lorenz equations
def xt(x, y, z, t):
    return sigma * (y - x)

def yt(x, y, z, t):
    return x * (rho - z) - y

def zt(x, y, z, t):
    return x * y - beta * z

# Runge-Kutta 4th order method
def rk4(t, dt, x, y, z):
    k1 = xt(x, y, z, t)
    l1 = yt(x, y, z, t)
    m1 = zt(x, y, z, t)

    k2 = xt(x + 0.5 * k1 * dt, y + 0.5 * l1 * dt, z + 0.5 * m1 * dt, t + dt / 2)
    l2 = yt(x + 0.5 * k1 * dt, y + 0.5 * l1 * dt, z + 0.5 * m1 * dt, t + dt / 2)
    m2 = zt(x + 0.5 * k1 * dt, y + 0.5 * l1 * dt, z + 0.5 * m1 * dt, t + dt / 2)

    k3 = xt(x + 0.5 * k2 * dt, y + 0.5 * l2 * dt, z + 0.5 * m2 * dt, t + dt / 2)
    l3 = yt(x + 0.5 * k2 * dt, y + 0.5 * l2 * dt, z + 0.5 * m2 * dt, t + dt / 2)
    m3 = zt(x + 0.5 * k2 * dt, y + 0.5 * l2 * dt, z + 0.5 * m2 * dt, t + dt / 2)

    k4 = xt(x + k3 * dt, y + l3 * dt, z + m3 * dt, t + dt)
    l4 = yt(x + k3 * dt, y + l3 * dt, z + m3 * dt, t + dt)
    m4 = zt(x + k3 * dt, y + l3 * dt, z + m3 * dt, t + dt)

    x_next = x + (dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    y_next = y + (dt * (l1 + 2 * l2 + 2 * l3 + l4) / 6)
    z_next = z + (dt * (m1 + 2 * m2 + 2 * m3 + m4) / 6)

    return x_next, y_next, z_next

# Parameters
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0
n = 5000
T = 50
dt = T / n

# Initial conditions
x0 = 10
y0 = 10
z0 = 10

# Initialize arrays
t_values = np.zeros(n + 1)
x_values = np.zeros(n + 1)
y_values = np.zeros(n + 1)
z_values = np.zeros(n + 1)

# Set initial values
t_values[0] = 0
x_values[0] = x0
y_values[0] = y0
z_values[0] = z0

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot initial point
line, = ax.plot([], [], [], 'r-', linewidth=0.7)

# Set axis limits
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)

# Set axis labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Text annotation for time
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    time_text.set_text('')
    return line, time_text

# Animation function. This is called sequentially
def animate(i):
    global x_values, y_values, z_values
    x_next, y_next, z_next = rk4(t_values[i], dt, x_values[i], y_values[i], z_values[i])
    t_values[i + 1] = t_values[i] + dt
    x_values[i + 1] = x_next
    y_values[i + 1] = y_next
    z_values[i + 1] = z_next
    
    line.set_data(x_values[:i], y_values[:i])
    line.set_3d_properties(z_values[:i])
    time_text.set_text('Time = {:.1f}'.format(t_values[i]))
    return line, time_text

# Call the animator. blit=True means only re-draw the parts that have changed.
# Call the animator. Set repeat=False to stop after one run.
ani = FuncAnimation(fig, animate, init_func=init, frames=n, interval=1, blit=True, repeat=False)


plt.show()

