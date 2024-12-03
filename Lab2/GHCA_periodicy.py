import numpy as np
import random
import matplotlib.pyplot as plt

# initialize hastings with 20x20 matrix size
n = 20
m = 20

C = np.empty([n,m], dtype=int)

N = 10
states = [0,N-1]

# excited states
e = [1,2]
threshold = 1

# randomly generate an initial configuration for GHCA.
for i in range(n):
  for j in range(m):
    C[i][j] = random.randint(states[0],states[1])

# returns the von-moore neighbours of a cell at i,j for matrix C
def neighbours(i,j,C):
    neighbours = []

    if max(i - 1, 0) != i:
        neighbours.append(C[max(i - 1, 0), j])

    if max(j - 1, 0) != j:
        neighbours.append(C[i, max(j - 1, 0)])

    if min(i + 1, len(C) - 1) != i:
        neighbours.append(C[min(i + 1, len(C) - 1)][j])

    if min(j + 1, len(C[i]) - 1) != j:
        neighbours.append(C[i][min(j + 1, len(C[i]) - 1)])

    return neighbours

# function counting the number of excited neighbours.
def count(neighbours, excited_cells):
    counts = 0
    for i in neighbours:
        if i in excited_cells:
            counts += 1
    return counts


# these 2 lists below store the configuration and respective
# time step for every time steps where (step+1) % N==0
period_mats = []
period_mats.append(C)
period_time = []

# GHCA applied 50 times giving 100 configurations
steps = 50
for step in range(steps):
    C_temp = np.empty([n, m], dtype=int)

    for i in range(n):
        for j in range(m):
            if 1<= C[i][j] <=N-2:
                C_temp[i][j] = C[i][j] + 1

            elif C[i][j]== N-1:
                C_temp[i][j] = 0

            else:
                counts = count(neighbours(i, j, C), e)
                if counts >= threshold:
                    C_temp[i][j] = 1
                else:
                    C_temp[i][j] = 0

    C = C_temp

    period_mats.append(C)

# the for loop below plots the non-transient orbits
for i in range(0,len(period_mats)):
    for j in range(i+1,len(period_mats)):
        if np.array_equal(period_mats[i],period_mats[j]):
            period_time.append((i,j))
            break

# Plot the first pair of matrices in period_time
if period_time:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for idx, index in enumerate(period_time[0]):
        ax = axes[idx]
        ax.imshow(period_mats[index], cmap='plasma', interpolation='nearest')
        ax.set_title(f"Step {index}")
        ax.set_xticks([])
        ax.set_yticks([])
else:
    print("No periodic configuration found.")

def compute_distance(config1, config2):
    return np.sum(config1 != config2)

orbit = []
for i in range(period_time[0][0], period_time[0][1]):
    orbit.append(period_mats[i])

#print(period_time)
C_p = np.empty([n,m], dtype=int)
C_p = period_mats[period_time[0][0]]

C = C_p.copy()

# Randomly modify a few cells
num_modifications = 5
for _ in range(num_modifications):
    i, j = random.randint(0, n-1), random.randint(0, m-1)
    C[i, j] = random.randint(0, N-1)

max_steps = 50
distances = []
for step in range(0,max_steps):
    new_C = np.empty_like(C)

    for i in range(n):
        for j in range(m):
            if 1<= C[i][j] <=N-2:
                new_C[i][j] = C[i][j] + 1

            elif C[i][j]== N-1:
                new_C[i][j] = 0

            else:
                counts = count(neighbours(i, j, C), e)
                if counts >= threshold:
                    new_C[i][j] = 1
                else:
                    new_C[i][j] = 0

    min_distaces = []
    for i in range(len(orbit)):
        distance = compute_distance(C,orbit[i])
        min_distaces.append(distance)

    min_dist = min(min_distaces)
    distances.append(min_dist)

    # Update C for the next iteration
    C = new_C.copy()

# Plot the distance over time
plt.figure(2)
plt.plot(range(len(distances)), distances)
plt.xlabel("Iteration")
plt.ylabel("Distance to Cp")
plt.title("Distance to Periodic Configuration Cp over Iterations")

plt.show()