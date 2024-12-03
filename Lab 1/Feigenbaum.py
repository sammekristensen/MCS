import numpy as np
from scipy.optimize import newton

def logistic_map(mu, x):
    return mu * x * (1 - x) 

def dlogistic_map(mu, x):
    return mu * (1 - 2 * x)

def sine_map(mu,x):
    return mu*np.sin(np.pi*x)

def sine_map_deriv(mu,x):
    return mu*np.pi*np.cos(np.pi*x)


def f_logistic(n, mu):
    logistic_val = np.zeros(2**n)
    logistic_val[0] = logistic_map(mu, 0.5)
    dlogistic_val = 0.25
    for i in range(1, 2**n):
        logistic_val[i] = logistic_map(mu, logistic_val[i-1])
        devpart = dlogistic_map(mu, logistic_val[i-1])
        dlogistic_val *= devpart
    return logistic_val[-1] - 0.5, dlogistic_val

def f_sin(n,mu):
    sine_val = np.zeros(2**n)
    sine_val[0] = sine_map(mu,0.5)
    sine_val_deriv = np.sin(np.pi*0.5)
    for i in range(1,2**n):
        sine_val[i] = sine_map(mu, sine_val[i-1])
        sine_devpart = sine_map_deriv(mu,sine_val[i-1])
        sine_val_deriv *= sine_devpart
    return sine_val[-1] -0.5, sine_val_deriv


def run_feigenbaum():
    iterations = 14
    deltareal = 4.6692016

    mu_values = np.zeros(iterations)
    mu_values[0] = 2
    mu_values[1] = 3.23607

    mu_sin_values = np.zeros(iterations)
    mu_sin_values[0] = 0.5
    mu_sin_values[1] = 0.7777

    print("feigenbaum for logistic:")
    print(" i        a_i        delta_i        alpha_i")
    for n in range(2, iterations):
        mu0 = mu_values[n-1] + (mu_values[n-1] - mu_values[n-2]) / deltareal
        mu_values[n] = newton(lambda mu: f_logistic(n, mu)[0], mu0, fprime=lambda mu: f_logistic(n, mu)[1],tol=1e-12 ,maxiter=10000)
        b0 = f_logistic(n-1,mu_values[n-1])[1]
        b1 = f_logistic(n,mu_values[n])[1]
        alpha = b1/b0
        d = (mu_values[n-1]-mu_values[n-2])/(mu_values[n]-mu_values[n-1])
        print("%2d    %1.8f    %1.8f    %1.8f" % (n, mu_values[n], d, alpha))

    print()
    print("feigenbaum for sin:")
    print(" i        a_i        delta_i        alpha_i")
    for n in range(2,iterations):
        mu0 = mu_sin_values[n-1] + (mu_sin_values[n-1] - mu_sin_values[n-2]) / deltareal
        mu_sin_values[n] = newton(lambda mu: f_sin(n, mu)[0], mu0, fprime=lambda mu: f_sin(n, mu)[1],tol=1e-12 ,maxiter=10000)
        d = (mu_sin_values[n-1]-mu_sin_values[n-2])/(mu_sin_values[n]-mu_sin_values[n-1])
        b0 = f_sin(n-1,mu_sin_values[n-1])[1]
        b1 = f_sin(n,mu_sin_values[n])[1]
        alpha = b1/b0
        print("%2d    %1.8f    %1.8f    %1.8f" % (n, mu_sin_values[n], d, alpha))

run_feigenbaum()