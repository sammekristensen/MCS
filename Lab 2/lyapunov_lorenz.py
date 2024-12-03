import numpy as np
import matplotlib.pyplot as plt
from Lorenz import rk4, xt, yt, zt

# Parameters
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0
n = 2000
T = 20
dt = T / n

epsilon = 1e-5

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

# Initial conditions perturbated
x0p = x0
y0p = y0
z0p = z0 + epsilon

# Initialize arrays perturbated
x_valuesp = np.zeros(n + 1)
y_valuesp = np.zeros(n + 1)
z_valuesp = np.zeros(n + 1)

# Set initial values perturbated
x_valuesp[0] = x0p
y_valuesp[0] = y0p
z_valuesp[0] = z0p


for i in range(n):
    x_next, y_next, z_next = rk4(t_values[i], dt, x_values[i], y_values[i], z_values[i])
    t_values[i + 1] = t_values[i] + dt
    x_values[i + 1] = x_next
    y_values[i + 1] = y_next
    z_values[i + 1] = z_next

    x_nextp, y_nextp, z_nextp = rk4(t_values[i], dt, x_valuesp[i], y_valuesp[i], z_valuesp[i])
    x_valuesp[i + 1] = x_nextp
    y_valuesp[i + 1] = y_nextp
    z_valuesp[i + 1] = z_nextp

def get_length(x, y, z, xp, yp, zp):
    # Calculate the difference in coordinates
    dx = xp - x
    dy = yp - y
    dz = zp - z
    
    # Calculate the Euclidean distance
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return length

delta_xt0 = epsilon
lambda_values = []
t_plot = []
for i in range(0, n+1, 100):
    t_plot.append(t_values[i])
    delta_xti = get_length(x_values[i],y_values[i],z_values[i], x_valuesp[i],y_valuesp[i],z_valuesp[i])
    lambda_values.append(np.log(np.abs(delta_xti)/delta_xt0))

slope, intercept = np.polyfit(t_plot[1:], lambda_values[1:], 1)    # fit line
d_array = [val*slope+intercept for val in t_plot[1:]]    # calculate line

plt.plot(t_plot[1:],lambda_values[1:], 'o', markersize=2)
plt.plot(t_plot[1:], d_array, '--', label=f'Lyapunov exponent = {slope:.2f}')
plt.title('Lyapunov for perturbation in z')
plt.xlabel('$t_{i}$')
plt.ylabel('$\lambda_{i}^{z}$')
# Set custom x-ticks
plt.xticks(np.arange(1, 21, 1))
plt.legend()

plt.show()
