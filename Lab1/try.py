import math

def q_mu(x, mu):
    return mu*x*(1-x)

def find_preimages(interval, mu, precision=0.0001):
    preimages = []
    for x in range(0, 10000):
        x_val = x * precision
        if interval[0] <= q_mu(x_val, mu) <= interval[1]:
            preimages.append(x_val)
    return preimages

def iterate_q(x, mu, n):
    for _ in range(n):
        x = q_mu(x, mu)
    return x

def find_period_3_orbit(mu):
    interval = (1/mu, (1 - math.sqrt(1 - 4/mu)) / 2)
    preimages = find_preimages(interval, mu)

    half = len(preimages)//2
    preimage1 = [preimages[0], preimages[half-1]]
    preimage2 = [preimages[half], preimages[-1]]
    print("Preimage 1 for the interval: ",preimage1, "Preimage 2 for the interval:", preimage2)
    return 

mu_value = 4
period_3_orbit = find_period_3_orbit(mu_value)
