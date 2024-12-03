from hexagon import *
import random
import matplotlib.pyplot as plt
import numpy as np

class Person:
    def __init__(self, status:str):
        self.status = status
    
    def update(self, infection_prob, recover_prob):
        if self.status == "susceptible":
            if random.random() < infection_prob:
                self.status = "infected"
        elif self.status == "infected":
            if random.random() < recover_prob:
                self.status = "recovered"
    
    def is_infected(self):
        return self.status == "infected"
    
class Cell:
    def __init__(self, population:int, infection_prob, recover_prob):
        self.population = population
        self.velocity_channel = [None] * 6
        self.temp_velocity_channel = [None]* 6
        self.infection_prob = infection_prob
        self.recover_prob = recover_prob

    
    def populate(self, prop_infected):
        for person in range(0,self.population):
            placed = False
            while not placed:
                location = random.randint(0,5)
                if not self.velocity_channel[location]:
                    self.velocity_channel[location] = Person("infected") if random.random() < prop_infected else Person("susceptible")
                    placed = True
        #self.velocity_channel = [Person("infected") if random.random() < prop_infected else Person("susceptible") for _ in range(self.population)]

    def contact_operation(self):
        num_infected = sum(1 for person in self.velocity_channel if person and person.is_infected())
        probability = 1 - (1 - self.infection_prob) ** num_infected
        for person in self.velocity_channel:
            if person:
                person.update(probability, self.recover_prob)
    
    def random_movement(self):
        random.shuffle(self.velocity_channel)

    def receive_new_person(self, person:Person, position):
        self.temp_velocity_channel[position] = person
    
    def switch_velocity_channel(self):
        self.velocity_channel = self.temp_velocity_channel
        self.temp_velocity_channel = [None]* 6

    def remove_person(self, person):
        for i in range(len(self.velocity_channel)):
            if self.velocity_channel[i] == person:
                self.velocity_channel[i] = None

    def calc_population(self):
        self.population = sum(1 for person in self.velocity_channel if person)

    def demographic(self):
        susceptible = sum(1 for person in self.velocity_channel if person and person.status == "susceptible")
        infected = sum(1 for person in self.velocity_channel if person and person.status == "infected")
        recovered = sum(1 for person in self.velocity_channel if person and person.status == "recovered")
        return susceptible, infected, recovered
    
class Hex_grid:
    def __init__(self, right:int, bottom:int, r_val,a) -> None:    
        total_population = 16100
        self.grid = {}
        for q in range(0, right):
            q_offset = math.floor(q/2.0)
            for r in range(0-q_offset, bottom - q_offset):
                population = random.randint(0, 6)
                self.grid[Hex(q,r,-q-r)] =  Cell(population,r_val,a)
                total_population -= population

        # Find max column, row in offset cords
        self.col = max(qoffset_from_cube(ODD, hex_key).col for hex_key in self.grid) + 1
        self.row = max(qoffset_from_cube(ODD, hex_key).row for hex_key in self.grid) + 1

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
        susceptible, infected, recovered = 0, 0, 0
        for cell_value in self.grid.values():
            sus, inf, rec = cell_value.demographic()
            susceptible += sus
            infected += inf
            recovered += rec
        return susceptible, infected, recovered
    
    def calc_population(self):
        return sum(cell_value.population for cell_value in self.grid.values())

    def populate_grid(self, prop_infected=0.006):
        for cell in self.grid.values():
            cell.populate(prop_infected)

def run_simulation(steps,r,a):
    grid = Hex_grid(100,100,r,a)
    grid.populate_grid()
    print("Initial population:", grid.calc_population())

    susceptible = np.empty(steps)
    infected = np.empty(steps)
    recovered = np.empty(steps)

    susceptible[0], infected[0], recovered[0] = grid.hex_grid_demographic()
    print(susceptible[0], infected[0], recovered[0])
    for i in range(steps):
        grid.propagate_step()
        susceptible[i], infected[i], recovered[i] = grid.hex_grid_demographic()

    return susceptible, infected, recovered

if __name__ == "__main__":
    steps = 100
    r = 0.3
    a = 0.2

    susceptible, infected, recovered = run_simulation(steps,r,a)

    x = np.linspace(0, steps, steps)
    plt.plot(x, infected, linewidth=1.5, color='red')
    plt.plot(x, susceptible, linewidth=1.5, color='blue')
    plt.plot(x, recovered, linewidth=1.5, color='green')
    plt.legend(["$N_{I}$", "$N_{S}$", "$N_{R}$"])
    plt.xlabel("Time step k")
    plt.ylabel("Number of $N_{I},N_{S},N_{R}$")
    plt.show()