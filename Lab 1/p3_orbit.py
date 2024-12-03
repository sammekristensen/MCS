import numpy as np
from scipy.optimize import fsolve

# Define the logistic map function q_mu
def q_mu(mu, x):
    return mu * x * (1 - x)

# Function to find preimages of the given interval under q_mu
def find_preimages(mu, p):
    # Calculate the bounds of the interval
    lower_bound = 1 / mu
    upper_bound = (1 - p / (1 - 4 / mu)) / 2
    
    # Define functions to solve
    def equation1(x):
        return q_mu(mu, x) - lower_bound
    
    def equation2(x):
        return q_mu(mu, x) - upper_bound
    
    # Find roots using fsolve
    preimage1 = fsolve(equation1, 0.5)[0]
    preimage2 = fsolve(equation2, 0.5)[0]
    
    return preimage1, preimage2

# Define parameters
mu = 3.5
p = 0.3

# Find the preimages of the interval under q_mu
preimage1, preimage2 = find_preimages(mu, p)
print("Preimage 1:", preimage1)
print("Preimage 2:", preimage2)

# Choose the correct preimage (choose one of them arbitrarily)
chosen_preimage = preimage2

# Iterate the chosen preimage three times under q_mu
def iterate_preimage(preimage, mu, n):
    for _ in range(n):
        preimage = q_mu(mu, preimage)
    return preimage

# Iterate the chosen preimage three times
iterated_preimage = iterate_preimage(chosen_preimage, mu, 3)

# Check if the iterated preimage covers itself
covers_itself = np.isclose(iterated_preimage, chosen_preimage)

# Print the result
print("Chosen Preimage:", chosen_preimage)
print("Iterated Preimage after 3 iterations:", iterated_preimage)
print("Covers Itself after 3 iterations:", covers_itself)




# Explanation of why the orbit is not a fixed point
# A fixed point under q_mu^3 means q_mu^3(x) = x for some x.
# However, in a period 3 orbit, q_mu^3(x) = x only if x is a fixed point itself,
# which contradicts the definition of a period 3 orbit being distinct points.