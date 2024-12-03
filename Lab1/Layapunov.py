import numpy as np
import matplotlib.pyplot as plt


def sine_map(mu, x):
    return mu*np.sin(np.pi*x)

def sine_map_deriv(mu, x):
    return mu*np.pi*np.cos(np.pi*x)

def lyapunov():
    mu_values = np.linspace(0, 1, 10000)
    lambda_values = []

    for mu in mu_values:
        x = np.random.random()
        for _ in range(1000):  # Discard the first 1000 iterations
            x = sine_map(mu, x)
        sum = 0
        for _ in range(1000):  # Next 10000 iterations
            x = sine_map(mu, x)
            sum += np.log(abs(sine_map_deriv(mu, x)))
        lambda_values.append(sum / 1000)


    plt.plot(mu_values, lambda_values, linewidth=0.7)
    plt.axhline(y=0, color='black', linestyle='--',linewidth=0.5)
    plt.xlabel('mu')
    plt.ylabel('lambda')
    plt.show()

lyapunov()

#This code calculates the Lyapunov exponent 
# for the logistic map for mu values between
# 2.4 and 4. The Lyapunov exponent is negative 
# when the system is stable (the trajectories
# converge) and positive when the system is chaotic 
# (the trajectories diverge). The periodic windows in the 
# bifurcation diagram correspond to regions where the system
# is stable, which is why the Lyapunov exponent is negative 
# there. The chaotic regions correspond to regions where the 
# system is chaotic, which is why the Lyapunov exponent is 
# positive there.