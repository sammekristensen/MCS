import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def logistic_map(r, x):
    return r * x * (1 - x)

def perturbed_logistic_map(r, x, a):
    return logistic_map(r, x) + a * x**4

def sine_map(r, x):
    return r * np.sin(np.pi * x)

interval = (0, 1)  # start, end
accuracy = 0.0001
reps = 600  # number of repetitions
numtoplot = 200
a = 0.1  # Small constant for perturbed logistic map
result = np.zeros(reps)

fig, biax = plt.subplots()
fig.set_size_inches(10, 6)

# Compute and plot bifurcation diagram for logistic map
result[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps - 1):
        result[i + 1] = logistic_map(r, result[i])

    biax.plot([r] * numtoplot, result[reps - numtoplot :], "r.", markersize=0.02,
              label='Logistic Map' if r == interval[0] else "")

# Compute and plot bifurcation diagram for perturbed logistic map
result[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps - 1):
        result[i + 1] = perturbed_logistic_map(r, result[i], a)

    biax.plot([r] * numtoplot, result[reps - numtoplot :], "b.", markersize=0.02,
              label='Perturbed Logistic Map' if r == interval[0] else "")

biax.set(xlabel="r", ylabel="x", title="Bifurcation Diagram of the Logistic and Perturbed Logistic Map")
legend_elements = [Line2D([0], [0], marker='o', color='r', markerfacecolor='r', markersize=10, label='Logistic Map'),
                   Line2D([0], [0], marker='o', color='b', markerfacecolor='b', markersize=10, label='Perturbed Logistic Map')]
biax.legend(handles=legend_elements)
plt.show()

# Compute and plot bifurcation diagram for sine map
fig, biax = plt.subplots()
fig.set_size_inches(10, 6)

result[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps - 1):
        result[i + 1] = sine_map(r, result[i])

    biax.plot([r] * numtoplot, result[reps - numtoplot :], "b.", markersize=0.02)

biax.set(xlabel="r", ylabel="x", title="Bifurcation Diagram of the Sine Map")
plt.show()
