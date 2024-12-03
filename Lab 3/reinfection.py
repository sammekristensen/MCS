from hexagon import *
import random
import matplotlib.pyplot as plt
import numpy as np

class Person:
    def __init__(self, status, k, l):
        """The status is if the person is S,I,R (or vaccinated), k for time untill vaccination
          gets recovered and l for time from recovered to susceptible"""
        self.status = status
        self.k = k
        self.l = l
    
    def update(self, infection_prob, recover_prob,l):
        """Uppdates the status based on infection and recovery probability, also the rules for k and l"""
        if self.status == "susceptible":
            if random.random() < infection_prob:
                self.status = "infected"
        elif self.status == "infected":
            if random.random() < recover_prob:
                self.status = "recovered"
                self.l = l
        elif self.status == "vaccinated":
            if random.random() < infection_prob:
                self.status = "infected"
            elif self.k == 0:
                self.status = "recovered"
                self.l = l
            else:
                self.k -=1
        elif self.status == "recovered":
            if self.l == 0:
                self.status = "susceptible"
            else:
                self.l -= 1
    
    def is_infected(self):
        """Returns True if person is infected"""
        return True if self.status == "infected" else False
    
class Cell:
    def __init__(self, population:int, infection_prob, recover_prob, rest_channels):
        """Get number of population in cell, sets up velocity channels and rest channels,
        also the infection probability and recover probability"""
        self.population = population
        self.rest_channels = rest_channels
        self.velocity_channel = [None] * 6
        self.temp_velocity_channel = [None] * 6
        self.rest_channel = [None] * self.rest_channels
        self.rest_indices = random.sample(range(6),self.rest_channels)

        self.infection_prob = infection_prob
        self.recover_prob = recover_prob

    
    def populate(self, percent_infected,k, percent_vaccinated, l):
        """Populates the cell with people randomly infected, vaccinated and susceptible based on percentage"""
        list1 = [x for x in range(6)]
        rest_index = 0
        for person in range(0,self.population):
            placed = False
            rand = random.random()
            if rand < percent_infected:
                status = "infected"
            elif rand < percent_infected + percent_vaccinated:
                status = "vaccinated"
            else:
                status = "susceptible"

            while not placed:
                location = random.sample(list1, 1)[0]
                list1.remove(location)
                if location not in self.rest_indices:
                    self.velocity_channel[location] = Person(status,k, l)
                    placed = True

                elif location in self.rest_indices:
                    location = rest_index
                    self.rest_channel[location] = Person(status,k, l)
                    rest_index += 1
                    placed = True

    def contact_operation(self,l):
        """The contact operation C, calculates the probability of getting infected at each cell"""
        total_population = self.velocity_channel+self.rest_channel
        num_infected = sum(1 for person in total_population if person and person.is_infected())
        probability = 1 - (1 - self.infection_prob) ** num_infected
        for person in total_population:
            if person:
                person.update(probability, self.recover_prob,l)
    
    def randomization_operator(self):
        """The randomization operator R, randomly moves people in the cells"""
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
                    location = rest_index 
                    self.rest_channel[rest_index] = person
                    rest_index +=1
                    placed = True

                else:
                    empty_slots = [index for index, val in enumerate(self.velocity_channel) if val is None]
                    if empty_slots:
                        velocity_index = random.choice(empty_slots)
                        self.velocity_channel[velocity_index] = person
                        placed = True


    def receive_new_person(self, person:Person, position):
        """Helper function to recieve a new person"""
        self.temp_velocity_channel[position] = person
    
    def svc(self):
        """Switches temp velocity with velocity channel"""
        self.velocity_channel = self.temp_velocity_channel
        self.temp_velocity_channel = [None] * 6

    def remove_person(self, person):
        """Removes a person"""
        for i in range(len(self.velocity_channel)):
            if self.velocity_channel[i] == person:
                self.velocity_channel[i] = None
        for i in range(len(self.rest_channel)):
            if self.rest_channel[i] == person:
                self.rest_channel[i] = None

    def calculate_population(self):
        """Calculates the whole population"""
        self.population = sum(1 for person in self.velocity_channel if person) + sum(1 for person in self.rest_channel if person)

    def demographic(self):
        """Calculates the demographics for each state"""
        susceptible = sum(1 for person in self.velocity_channel if person and person.status == "susceptible") + sum(1 for person in self.rest_channel if person and person.status == "susceptible")
        infected = sum(1 for person in self.velocity_channel if person and person.status == "infected") + sum(1 for person in self.rest_channel if person and person.status == "infected")
        recovered = sum(1 for person in self.velocity_channel if person and person.status == "recovered") + sum(1 for person in self.rest_channel if person and person.status == "recovered")
        vaccinated = sum(1 for person in self.velocity_channel if person and person.status == "vaccinated") + sum(1 for person in self.rest_channel if person and person.status == "vaccinated")
        return susceptible, infected, recovered, vaccinated
    
class Hexagon:
    def __init__(self, right, bottom, r_val, a, rest_channels,k,l):
        """Creates the hexagon grid, finds max column, row in offset coordinates"""
        self.grid = {}
        for q in range(0, right):
            q_offset = math.floor(q/2.0)
            for r in range(0-q_offset, bottom - q_offset):
                population = random.randint(0, 6)
                self.grid[Hex(q,r,-q-r)] =  Cell(population, r_val, a, rest_channels)

        self.col = max(qoffset_from_cube(ODD, hex_key).col for hex_key in self.grid) + 1
        self.row = max(qoffset_from_cube(ODD, hex_key).row for hex_key in self.grid) + 1
        self.rest_channels = rest_channels
        self.k = k
        self.l = l

    def run_step(self):
        """Next step in the simulation"""
        for hex_key, cell_value in self.grid.items():
            cell_value.contact_operation(self.l)
            cell_value.randomization_operator()
            for i, person in enumerate(cell_value.velocity_channel):
                if person:
                    self.switch_person(hex_key, person, i)

        for cell_value in self.grid.values():
            cell_value.svc()

    def switch_person(self, curr_hex, person, direction):
        """Switches the person with the neighbouring cell"""
        neighbouring_hexagon = hex_neighbor(curr_hex, direction)
        neighbour_offset_cord = qoffset_from_cube(ODD, neighbouring_hexagon)
        neighbouring_hexagon = qoffset_wraparound_to_hex(neighbour_offset_cord, self.col, self.row)
        neighbour_cell = self.grid[neighbouring_hexagon]
        neighbour_cell.receive_new_person(person, direction)

    def total_deomgraphic(self):
        """Calculates the whole demographic for the hexagon grid"""
        susceptible, infected, recovered, vaccinated = 0, 0, 0, 0
        for cell_value in self.grid.values():
            sus, inf, rec, vac = cell_value.demographic()
            susceptible += sus
            infected += inf
            recovered += rec
            vaccinated += vac
        return susceptible, infected, recovered, vaccinated
    
    def calculate_population(self):
        """Calculates the total population"""
        return sum(cell_value.population for cell_value in self.grid.values())

    def populate_grid(self, percent_infected,percent_vaccinated):
        """Populates the whole grid"""
        for cell in self.grid.values():
            cell.populate(percent_infected,self.k,percent_vaccinated, self.l)

def run_simulation(steps, r, a, rest_channels,k,l, percent_infected, percent_vaccinated):
    """Runs the simulation with the parameters given"""
    grid = Hexagon(100,100,r,a, rest_channels,k,l)
    grid.populate_grid(percent_infected, percent_vaccinated)
    print("Initial population:", grid.calculate_population())

    susceptible = np.empty(steps)
    infected = np.empty(steps)
    recovered = np.empty(steps)
    vaccinated = np.empty(steps)

    susceptible[0], infected[0], recovered[0], vaccinated[0] = grid.total_deomgraphic()
    for i in range(steps):
        grid.run_step()
        susceptible[i], infected[i], recovered[i], vaccinated[i] = grid.total_deomgraphic()

    return susceptible, infected, recovered, vaccinated

def plot_fig(steps,susceptible,infected,recovered, vaccinated):
    """Plots the figure"""
    x = np.linspace(0, steps, steps)
    plt.plot(x, infected, linewidth=1.5, color='red')
    plt.plot(x, susceptible, linewidth=1.5, color='blue')
    plt.plot(x, recovered, linewidth=1.5, color='green')
    plt.plot(x, vaccinated, linewidth=1.5, color='black')
    plt.legend(["$N_{I}$", "$N_{S}$", "$N_{R}$", "$N_{V}$"])
    plt.xlabel("Time step k")
    plt.ylabel("Number of $N_{I},N_{S},N_{R}, N_{V}$")
    plt.show()

if __name__ == "__main__":
    """Set parameters here"""
    steps = 100
    r = 0.3
    a = 0.2
    rest_channels = 0
    k = 10
    l = 50
    percent_infected = 0.005
    percent_vaccinated = 0.1

    susceptible, infected, recovered, vaccinated = run_simulation(steps,r,a,rest_channels,k,l,percent_infected, percent_vaccinated)

    plot_fig(steps,susceptible,infected,recovered, vaccinated)