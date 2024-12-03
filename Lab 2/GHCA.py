import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# initialize hastings with 20x20 matrix size
n = 20
m = 20

C = np.empty([n,m], dtype=int)

N = 10      # Number of states
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

# GHCA applied 100 times giving 100 configurations
steps =50
# Create initial plot with color bar

fig, ax = plt.subplots()
cax = ax.imshow(C, cmap='plasma', interpolation='nearest')
cbar = fig.colorbar(cax)

# Function to update the plot for each step
def update(frame):
    global C
    # Update C
    C_temp = np.empty_like(C)  # Create empty matrix to store updated values
    
    for i in range(len(C)):
        for j in range(len(C[i])):
            if 1 <= C[i][j] <= N - 2:
                C_temp[i][j] = C[i][j] + 1
            elif C[i][j] == N - 1:
                C_temp[i][j] = 0
            else:
                counts = count(neighbours(i, j, C), e)
                if counts >= threshold:
                    C_temp[i][j] = 1
                else:
                    C_temp[i][j] = 0
    
    C = C_temp

    # Update plot
    cax.set_data(C)
    plt.title(f"Step {frame + 1}")
    plt.pause(0.1)

# Create animation
animation = FuncAnimation(fig, update, frames=steps, repeat=False)

plt.show()

# save the final matrix in final_matrice.txt
fin_matr = open('final_matrice.txt','a')
fin_matr.writelines([str(C[i,j])+' ' if j<m-1  else str(C[i,j])+'\n' for i in range(len(C)) for j in range(len(C[i]))])

fin_matr.close()
