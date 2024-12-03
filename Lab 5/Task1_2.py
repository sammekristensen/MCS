import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

NUM_BOIDS = 100
XLIM = 100
YLIM = 100
R1 = 10
R2 = 20
R3 = 30


class Boid:
    def __init__(self):
        self.x = random.uniform(-XLIM/2, XLIM/2)
        self.y = random.uniform(-YLIM/2, YLIM/2)
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)

        self.repell = [] # Zone 1
        self.align  = [] # Zone 2
        self.attract  = [] # Zone 3

    def update(self, delta_t):
        rho1 = 1
        rho2 = 0
        rho3 = 0
        rho4 = 0
        alpha = 0.8
        beta = 0.2

        self.vx = rho1*self.repell_vec()[0] + rho2*self.align_vec()[0] + rho3*self.attract_vec()[0]  + rho4*self.vx
        self.vy = rho1*self.repell_vec()[1] + rho2*self.align_vec()[1] + rho3*self.attract_vec()[1] + rho4*self.vy

        self.x += alpha*self.vx*delta_t + beta*self.brownian_motion(delta_t)[0]
        self.y += self.vy*delta_t + beta*self.brownian_motion(delta_t)[1]

        # Apply periodic boundary conditions
        self.x = (self.x + XLIM) % (2 * XLIM) - XLIM
        self.y = (self.y + YLIM) % (2 * YLIM) - YLIM

    def brownian_motion(self,delta_t):
        c = 3
        r_max = 10
        a = 1/(1-np.e**(-r_max**2/(2*c**4*delta_t**2)))
        u = random.uniform(0,1)
        R = np.sqrt(-2*c**4*delta_t**2*np.log((a-u)/a))
        theta = random.uniform(0,2*np.pi)
        return(R*np.cos(theta),R*np.sin(theta))
    
    def influence(self, particles):
        for p in particles:
            distance = np.sqrt((self.x - p.x)**2 + (self.y- p.y)**2)
            if distance < R1:
                self.repell += [p]
            elif distance < R2:
                self.align += [p]
            elif distance < R3:
                self.attract += [p]

    def center_of_mass(self,parts):
        if len(parts) == 0:
            return (0,0)
        x_center = 0
        y_center = 0
        for p in parts:
            x_center += p.x
            y_center += p.y
        center = (x_center/len(parts), y_center/len(parts)) 
        return center

    def repell_vec(self):
        #compute the normalized vector from the center of mass to the particle    
        center = self.center_of_mass(self.repell)
        delta_x = self.x - center[0]
        delta_y = self.y - center[1]
        theta_1 = np.arctan2(delta_y, delta_x)

        e_1 = (np.cos(theta_1), np.sin(theta_1))
        return e_1
    
    def align_vec(self):
        # Instead of e_2 we have v_2, i.e. average velocity of particles in zone 2
        count = 0
        vel_x = 0
        vel_y = 0
        for p in self.align:
            vel_x += p.vx
            vel_y += p.vy
            count += 1
        if count == 0:
            return (0,0)
        v_2 = (vel_x /len(self.align), vel_y /len(self.align))
        return v_2

    def attract_vec(self):
        center=self.center_of_mass(self.attract)

        delta_x = self.x - center[0]
        delta_y = self.y - center[1]
        theta_3 = np.arctan2(delta_y, delta_x)
        # Compute the normalized vector towards the center of mass
        e_3 = (-np.cos(theta_3), -np.sin(theta_3))
        return e_3

class Simulation:
    def __init__(self,steps,particles):
        self.steps = steps
        self.particles = particles
        self.fig, self.ax =plt.subplots()
        self.ax.set_xlim(-XLIM, XLIM)
        self.ax.set_ylim(-YLIM, YLIM)
        self.scatter = self.ax.scatter([],[])
    def initial_animation(self):
        self.scatter.set_offsets([0,0])
        return self.scatter,
    
    def update(self,frame):
        for p in self.particles:
            p.influence(self.particles)
            p.update(1)
        self.scatter.set_offsets([[p.x,p.y] for p in self.particles])
        return self.scatter,
        
    def animate(self):
        anim = animation.FuncAnimation(self.fig,self.update,frames=self.steps,init_func=self.initial_animation, blit=True)
        plt.show()

if __name__ == "__main__":
    particles = [Boid() for _ in range(NUM_BOIDS)]
 
    simulation = Simulation(50,particles)
    simulation.animate()