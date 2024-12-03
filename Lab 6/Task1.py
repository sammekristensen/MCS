import numpy as np
import random
import matplotlib.pyplot as plt
from painter_play import painter_play
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

PLOT = 1
ANIMATE = 0

def initialize_population(pop_size, chromosome_length):
    '''Initializes the population'''
    return np.random.randint(0, 4, (pop_size, chromosome_length))

def evaluate_population(population, room):
    '''Evaluates the population, takes the average fitness for 5 tests for each cromosone'''
    nr_tests = 5
    fitness = np.zeros(population.shape[0])
    for i, chromosome in enumerate(population):
        tot_score = 0
        for _ in range(nr_tests):
            score, _, _ = painter_play(chromosome, room)
            tot_score += score
        fitness[i] = tot_score/nr_tests
    return fitness

def select_parents(population, fitness):
    '''Selects the parents based on probability, higher fitness gets higher probability'''
    idx = np.random.choice(np.arange(len(population)), size=len(population), p=fitness/fitness.sum())
    return population[idx]

def crossover(parents, crossover_rate=0.5):
    '''Does a 2 point crossovers for "crossover_rate" of the population, the rest reamins the same"'''
    offspring = np.empty(parents.shape)
    for k in range(0, len(parents), 2):
        parent1, parent2 = parents[k], parents[k+1]
        if np.random.rand() < crossover_rate:
            pt1, pt2 = sorted(random.sample(range(len(parent1)), 2))
            offspring[k, :pt1] = parent1[:pt1]
            offspring[k, pt1:pt2] = parent2[pt1:pt2]
            offspring[k, pt2:] = parent1[pt2:]
            offspring[k+1, :pt1] = parent2[:pt1]
            offspring[k+1, pt1:pt2] = parent1[pt1:pt2]
            offspring[k+1, pt2:] = parent2[pt2:]
        else:
            offspring[k], offspring[k+1] = parent1, parent2
    return offspring

def mutate(offspring, mutation_rate=0.005):
    '''Mutates the population with a 0.005 mutation rate per locus'''
    for chromosome in offspring:
        for i in range(len(chromosome)):
            if np.random.rand() < mutation_rate:
                chromosome[i] = np.random.randint(0, 4)
    return offspring

def genetic_algorithm(pop_size, chromosome_length, generations, room):
    '''Runs the generic algorithm, initalizes population and simulates for given number of generations'''
    population = initialize_population(pop_size, chromosome_length)
    average_fitness_over_generations = []

    for i in range(generations):
        print("Gen " + str(i))
        fitness = evaluate_population(population, room)
        print("Average fitness: " + str(np.average(fitness)))
        average_fitness_over_generations.append(np.average(fitness))

        parents = select_parents(population, fitness)
        offspring = crossover(parents)
        population = mutate(offspring)

    best_chromosome = population[np.argmax(fitness)]
    return best_chromosome, average_fitness_over_generations, population

# Parameters
pop_size = 100
chromosome_length = 54
generations = 200
room = np.zeros((30, 60))

# Run Genetic Algorithm
best_chromosome, fitness_over_time, population = genetic_algorithm(pop_size, chromosome_length, generations, room)

if PLOT:
    # Plot fitness over generations
    plt.plot(fitness_over_time)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness Over Generations')
    plt.show()

    # Plot example trajectory of the best chromosome
    score, xpos, ypos = painter_play(best_chromosome, room)

    plt.plot(xpos, ypos)
    plt.plot(xpos[0], ypos[0], 'o', color='blue', label='start')
    plt.plot(xpos[-1], ypos[-1], 'o', color='red', label='end')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory of Best Chromosome, score = {:.3f}'.format(score))
    plt.legend()

    fig, ax = plt.subplots()

    # Plot the chromosomes
    ax.imshow(population, cmap='tab10', aspect='auto')

    # Set labels and title
    ax.set_xlabel('Locus')
    ax.set_ylabel('Chromosome')
    ax.set_title('Set of Chromosomes')

    # Create custom legend
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Colors in 'tab10' colormap for values 0, 1, 2, 3
    labels = ['No turn', 'Turn left', 'Turn right', 'Random turn']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=patches, title='Actions', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjust layout to fit the legend
    plt.show()

if ANIMATE:

    score, xpos, ypos = painter_play(best_chromosome, room)
    # Create an animation of the trajectory
    fig, ax = plt.subplots()
    ax.set_xlim(0, room.shape[0]+1)
    ax.set_ylim(0, room.shape[1]+1)
    line, = ax.plot([], [], lw=2)
    start_point, = ax.plot([], [], 'o', color='blue', label='start')
    end_point, = ax.plot([], [], 'o', color='red', label='end')

    def init():
        line.set_data([], [])
        start_point.set_data([], [])
        end_point.set_data([], [])
        return line, start_point, end_point

    def update(frame):
        line.set_data(xpos[:frame], ypos[:frame])
        start_point.set_data(xpos[0], ypos[0])
        end_point.set_data(xpos[frame-1], ypos[frame-1])
        return line, start_point, end_point

    ani = FuncAnimation(fig, update, frames=len(xpos), init_func=init, blit=True, repeat=False, interval=100)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory of Best Chromosome, score = {:.3f}'.format(score))
    plt.legend()
    plt.show()
