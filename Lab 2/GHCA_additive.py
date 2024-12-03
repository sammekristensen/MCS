import numpy as np
import random
import matplotlib.pyplot as plt

# randomly generate an initial configuration for GHCA.
def random_matrix():
    C = np.empty([n,m], dtype=int)
    for i in range(n):
        for j in range(m):
            C[i][j] = random.randint(states[0],states[1])
    return C

def add_matrix(C1,C2,n,m):
  C12 = np.empty([n,m], dtype=int)
  for i in range(n):
    for j in range(m):
      C12[i,j] = (C1[i,j] + C2[i,j])%N
  return C12

# returns the von-moore neighbours of a cell at i,j for matrix matr
def neighbours(i,j,matr):
    neighbours = []

    if max(i - 1, 0) != i:
        neighbours.append(matr[max(i - 1, 0), j])

    if max(j - 1, 0) != j:
        neighbours.append(matr[i, max(j - 1, 0)])

    if min(i + 1, len(matr) - 1) != i:
        neighbours.append(matr[min(i + 1, len(matr) - 1)][j])

    if min(j + 1, len(matr[i]) - 1) != j:
        neighbours.append(matr[i][min(j + 1, len(matr[i]) - 1)])

    return neighbours

# function counting the number of excited neighbours.
def count(neighbours, excited_cells):
    cnt = 0
    for i in neighbours:
        if i in excited_cells:
            cnt += 1
    return cnt


def GHCA(matr):
    # GHCA applied 50 times giving 100 configurations
    steps = 100
    for step in range(steps):
        tmatr = np.empty([n, m], dtype=int)

        for i in range(n):
            for j in range(m):
                if 1<= matr[i][j] <=N-2:
                    tmatr[i][j] = matr[i][j] + 1

                elif matr[i][j]== N-1:
                    tmatr[i][j] = 0

                else:
                    cnt = count(neighbours(i, j, matr), e)
                    if cnt >= threshold:
                        tmatr[i][j] = 1
                    else:
                        tmatr[i][j] = 0

        matr = tmatr
    return matr


for i in range(10):
    # initialize hastings with 20x20 matrix size
    n = 20
    m = 20

    N = 10
    states = [0,N-1]

    # excited states
    e = [1,2]
    threshold = 1


    C1 = random_matrix()
    C2 = random_matrix()
    C12 = add_matrix(C1,C2,n,m)

    FC1 = GHCA(C1)
    FC2 = GHCA(C2)

    FC12 = GHCA(C12)

    FC1_FC2 = add_matrix(C1,C2,n,m)

    print(np.array_equal(FC1_FC2,FC12))