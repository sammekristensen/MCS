from hexagon import *
import random
import matplotlib.pyplot as plt
import numpy as np

class Person:
    def __init__(self, status:str, k):
        self.status = status
        self.k = k
    
    def update(self, infection_prob, recover_prob):
        if self.status == "susceptible":
            if random.random() < infection_prob:
                self.status = "infected"
        elif self.status == "infected":
            if random.random() < recover_prob:
                self.status = "recovered"
        elif self.status == "vaccinated":
            if random.random() < infection_prob:
                self.status = "infected"
            elif self.k == 0:
                self.status = "recovered"
            else:
                self.k -=1
    
    def is_infected(self):
        return True if self.status == "infected" else False
    
class Cell:
    def __init__(self, population:int, infection_prob, recover_prob, rest_channels):
        self.population = population
        self.rest_channels = rest_channels
        self.velocity_channel = [None] * 6
        self.temp_velocity_channel = [None] * 6
        self.rest_channel = [None] * self.rest_channels
        self.rest_indices = random.sample(range(6),self.rest_channels)
        self.velocity_indices = [item for item in range(6) if item not in self.rest_indices]

        self.infection_prob = infection_prob
        self.recover_prob = recover_prob

    
    def populate(self, prop_infected,k, prop_vaccinated):
        list1 = [x for x in range(6)]
        rest_index = 0
        for person in range(0,self.population):
            placed = False
            rand = random.random()
            if rand < prop_infected:
                status = "infected"
            elif rand < prop_infected + prop_vaccinated:
                status = "vaccinated"
            else:
                status = "susceptible"
            while not placed:
                location = random.sample(list1, 1)[0]
                list1.remove(location)
                if location not in self.rest_indices:
                    self.velocity_channel[location] = Person(status,k)
                    placed = True

                elif location in self.rest_indices:
                    location = rest_index
                    self.rest_channel[location] = Person(status,k)
                    rest_index += 1
                    placed = True

    def contact_operation(self):
        total_population = self.velocity_channel+self.rest_channel
        num_infected = sum(1 for person in total_population if person and person.is_infected())
        probability = 1 - (1 - self.infection_prob) ** num_infected
        for person in total_population:
            if person:
                person.update(probability, self.recover_prob)
    
    def random_movement(self):
        total_population = self.velocity_channel + self.rest_channel
        #random.shuffle(total_population)
        total_population = [person for person in total_population if person is not None]

        self.rest_channel = [None]*self.rest_channels
        self.velocity_channel = [None]* 6

        list1 = [x for x in range(len(total_population))]
        rest_index = 0
        for person in total_population:
            placed = False
            while not placed:
                location = random.sample(list1, 1)[0]
                list1.remove(location)

                if location in self.rest_indices:

                    location = rest_index  # Find the index of the first empty slot
                    self.rest_channel[rest_index] = person  # Place the person in the empty slot
                    rest_index +=1
                    placed = True  # Set placed to True to exit the loop

                else:
                    # If the slot index is not in rest_indices, place the person in velocity_channel
                    empty_slots = [index for index, val in enumerate(self.velocity_channel) if val is None]
                    if empty_slots:
                        velocity_index = random.choice(empty_slots)  # Randomly select an empty slot
                        self.velocity_channel[velocity_index] = person  # Place the person in the empty slot
                        placed = True  # Set placed to True to exit the loop


    def receive_new_person(self, person:Person, position):
        self.temp_velocity_channel[position] = person
    
    def switch_velocity_channel(self):
        self.velocity_channel = self.temp_velocity_channel
        self.temp_velocity_channel = [None] * 6

    def remove_person(self, person):
        for i in range(len(self.velocity_channel)):
            if self.velocity_channel[i] == person:
                self.velocity_channel[i] = None
        for i in range(len(self.rest_channel)):
            if self.rest_channel[i] == person:
                self.rest_channel[i] = None

    def calc_population(self):
        self.population = sum(1 for person in self.velocity_channel if person) + sum(1 for person in self.rest_channel if person)

    def demographic(self):
        susceptible = sum(1 for person in self.velocity_channel if person and person.status == "susceptible") + sum(1 for person in self.rest_channel if person and person.status == "susceptible")
        infected = sum(1 for person in self.velocity_channel if person and person.status == "infected") + sum(1 for person in self.rest_channel if person and person.status == "infected")
        recovered = sum(1 for person in self.velocity_channel if person and person.status == "recovered") + sum(1 for person in self.rest_channel if person and person.status == "recovered")
        vaccinated = sum(1 for person in self.velocity_channel if person and person.status == "vaccinated") + sum(1 for person in self.rest_channel if person and person.status == "vaccinated")
        return susceptible, infected, recovered, vaccinated
    
class Hex_grid:
    def __init__(self, right:int, bottom:int, r_val, a, rest_channels,k) -> None:    
        self.grid = {}
        for q in range(0, right):
            q_offset = math.floor(q/2.0)
            for r in range(0-q_offset, bottom - q_offset):
                population = random.randint(0, 6)
                self.grid[Hex(q,r,-q-r)] =  Cell(population, r_val, a, rest_channels)

        # Find max column, row in offset cords
        self.col = max(qoffset_from_cube(ODD, hex_key).col for hex_key in self.grid) + 1
        self.row = max(qoffset_from_cube(ODD, hex_key).row for hex_key in self.grid) + 1
        self.rest_channels = rest_channels
        self.k = k

    def propagate_step(self):
        for hex_key, cell_value in self.grid.items():
            cell_value.contact_operation() # Infect, recover, etc
            cell_value.random_movement() # random_movement
            for i, person in enumerate(cell_value.velocity_channel):
                if person:
                    self.transfer_person(hex_key, person, i)

        # Switch vel_channels
        for cell_value in self.grid.values():
            cell_value.switch_velocity_channel()

    def transfer_person(self, curr_hex, person:Person, direction:int):
        # Find corresponding cell
        neighbour_hex = hex_neighbor(curr_hex, direction)
        neighbour_offset_cord = qoffset_from_cube(ODD, neighbour_hex)
        neighbour_hex = qoffset_wraparound_to_hex(neighbour_offset_cord, self.col, self.row)
        neighbour_cell = self.grid[neighbour_hex]
        neighbour_cell.receive_new_person(person, direction)

    def hex_grid_demographic(self):
        susceptible, infected, recovered, vaccinated = 0, 0, 0, 0
        for cell_value in self.grid.values():
            sus, inf, rec, vac = cell_value.demographic()
            susceptible += sus
            infected += inf
            recovered += rec
            vaccinated += vac
        return susceptible, infected, recovered, vaccinated
    
    def calc_population(self):
        return sum(cell_value.population for cell_value in self.grid.values())

    def populate_grid(self, prop_infected=0.005,prop_vaccinated = 0):
        for cell in self.grid.values():
            cell.populate(prop_infected,self.k,prop_vaccinated)

def run_simulation(steps, r, a, rest_channels,k):
    grid = Hex_grid(100,100,r,a, rest_channels,k)
    grid.populate_grid()
    print("Initial population:", grid.calc_population())

    susceptible = np.empty(steps)
    infected = np.empty(steps)
    recovered = np.empty(steps)
    vaccinated = np.empty(steps)

    susceptible[0], infected[0], recovered[0], vaccinated[0] = grid.hex_grid_demographic()
    print(susceptible[0], infected[0], recovered[0], vaccinated[0])
    for i in range(steps):
        print(i)
        grid.propagate_step()
        susceptible[i], infected[i], recovered[i], vaccinated[i] = grid.hex_grid_demographic()

    return susceptible, infected, recovered, vaccinated

def plot_fig(steps,susceptible,infected,recovered, vaccinated):
    x = np.linspace(0, steps, steps)
    plt.plot(x, infected, linewidth=1.5, color='red')
    plt.plot(x, susceptible, linewidth=1.5, color='blue')
    plt.plot(x, recovered, linewidth=1.5, color='green')
    plt.plot(x, vaccinated, linewidth=1.5, color='black')
    plt.legend(["$N_{I}$", "$N_{S}$", "$N_{R}$", "$N_{V}$"])
    plt.xlabel("Time step k")
    plt.ylabel("Number of $N_{I},N_{S},N_{R},N_{V}$")
    plt.show()

if __name__ == "__main__":
    steps = 100
    r = 0.3
    a = 0.2
    rest_channels = 0
    k = 10

    susceptible, infected, recovered, vaccinated = run_simulation(steps,r,a,rest_channels,k)

    plot_fig(steps,susceptible,infected,recovered, vaccinated)