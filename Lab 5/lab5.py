import pygame
from sys import exit
from random import randint, uniform, choice
import numpy as np
import cv2
import matplotlib.pyplot as plt

fps = 20

black = (0,0,0)
white = (255,255,255)

#for display
pygame.init()

# Get the display info
info = pygame.display.Info()

# Set the screen width and height to the display size
WIDTH, HEIGHT = info.current_w * 0.9, info.current_h * 0.8

screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("BOIDs")
clock = pygame.time.Clock()

def draw_text(surf,text,size,x,y,color=white):
    font_name=pygame.font.match_font('aerial')
    font=pygame.font.Font(font_name,size)
    text_surface=font.render(text,True,color)
    text_rect=text_surface.get_rect()
    text_rect.center=(x,y)
    surf.blit(text_surface,text_rect)

MAX_SPEED = 50
BOID_COLOR = (255,255,56)
SPECIAL_BOID_COLOR = (255, 10 ,10)
BOID_SIZE = 5

R1 = 20
R2 = 50
R3 = 100

vec = pygame.math.Vector2
#MAIN CLASS BOIDS
class Boid(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((BOID_SIZE,BOID_SIZE))
        self.image.fill(BOID_COLOR)
        self.rect = self.image.get_rect()
        self.pos = vec(randint(0, WIDTH), randint(0, HEIGHT))
        self.vel = vec(choice([-MAX_SPEED, MAX_SPEED]),choice([-MAX_SPEED, MAX_SPEED])).rotate(uniform(0, 360))
        self.rect.center = self.pos

        self.repell = [] # Zone 1
        self.align  = [] # Zone 2
        self.attract  = [] # Zone 3

    # Uppdate function
    def update(self, DELTA_TIME,beta):
        # Variables
        rho1 = 0.3  # repell factor
        rho2 = 0.3  # align factor
        rho3 = 0.3  # attract factor
        rho4 = 0.1  # same dir factor
        alpha = 1-beta   # influence of speed
        beta = beta   # influence of brownian motion

        # Update influence groups 
        self.influence(boids)

        # Uppdate velocity (*MAX_SPEED otherwise it is to slow)
        if not self.align and not self.repell and not self.attract:
            self.vel = self.vel
        else:
            self.vel = (rho1*self.repell_vec() + rho2*self.align_vec() + rho3*self.attract_vec() + rho4*self.vel) * MAX_SPEED

        # Ensures that all speeds are constants
        # not sure why this was needed since all vectors are normalized and rhos add up to 1 but through experiments i needed it
        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)

        # Update position with brownian motion
        self.pos += alpha*self.vel*DELTA_TIME + beta*self.brownian_motion(DELTA_TIME)

        # Create the boundary condition
        self.wrap_position(self.pos)
        self.rect.center = self.pos

    # Brownian motion function
    def brownian_motion(self, delta_t):
        c = MAX_SPEED/4
        r_max = 30  # Choose your maximum value of |X(∆t)| here
        a = 1 / (1 - np.exp(-r_max ** 2 / (2 * c ** 4 * delta_t ** 2)))
        u = uniform(0, 1)
        R = np.sqrt(-2 * c ** 4 * delta_t ** 2 * np.log((a - u) / a))

        # Truncate the distribution if R exceeds r_max
        if R > r_max:
            R = r_max

        # Raise the distribution to compensate for truncation
        # Ensure the area under the curve remains equal to 1
        f_R = np.exp(-R ** 2 / (2 * c ** 4 * delta_t ** 2))

        theta = uniform(0, 2 * np.pi)
        return vec(R * np.cos(theta), R * np.sin(theta)) * f_R

    # Funtion to divide  other boid to the groups
    def influence(self, boids):
        # Reset the lists to empty
        self.repell = []
        self.align = []
        self.attract = []
        for b in boids:
            if b != self:
                distance = self.wrap_distance(self.pos, b.pos)
                if distance.length() < R1:
                    self.repell += [b]
                elif distance.length() < R2:
                    self.align += [b]
                elif distance.length() < R3:
                    self.attract += [b]

    def wrap_position(self, pos):
        if pos.x >= WIDTH:
            pos.x -= WIDTH
        elif pos.x < 0:
            pos.x += WIDTH
        if pos.y >= HEIGHT:
            pos.y -= HEIGHT
        elif pos.y < 0:
            pos.y += HEIGHT
        return pos

    def wrap_distance(self, pos1, pos2):
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y

        if abs(dx) > WIDTH / 2:
            if dx > 0:
                dx -= WIDTH
            else:
                dx += WIDTH

        if abs(dy) > HEIGHT / 2:
            if dy > 0:
                dy -= HEIGHT
            else:
                dy += HEIGHT

        return vec(dx, dy)

    def center_of_mass(self, part):
        if len(part) == 0:
            return vec(0, 0)

        sum_x = sum_y = 0
        base_x, base_y = part[0].pos.x, part[0].pos.y

        if len(part) == 1:
            return vec(base_x, base_y)

        for particle in part:
            dx = particle.pos.x - base_x
            dy = particle.pos.y - base_y

            if abs(dx) > WIDTH / 2:
                if dx > 0:
                    dx -= WIDTH
                else:
                    dx += WIDTH

            if abs(dy) > HEIGHT / 2:
                if dy > 0:
                    dy -= HEIGHT
                else:
                    dy += HEIGHT

            sum_x += base_x + dx
            sum_y += base_y + dy

        center = vec(sum_x / len(part), sum_y / len(part))
        return self.wrap_position(center)

    def repell_vec(self):
        center = self.center_of_mass(self.repell)
        if center == vec(0, 0):
            return vec(0, 0)

        direction = self.wrap_distance(self.pos, center)
        if direction.length() > 0:
            direction = direction.normalize()
        return -direction
    
     # Function for allignment use average v_2(t)
    def align_vec(self):
        if len(self.align) == 0:
            return vec(0, 0)  # No alignment needed if no boids to align with
        
        # Calculate the sum of velocities
        sum_vel = vec(0, 0)
        for boid in self.align:
            sum_vel += boid.vel      
        # Calculate the average direction
        avg_direction = sum_vel.normalize()
        
        return avg_direction

    def attract_vec(self):
        center = self.center_of_mass(self.attract)
        direction = self.wrap_distance(self.pos, center)
        if direction.length() > 0:
            direction = direction.normalize()
        return direction
    
def compute_order_parameter(boids):
    velocity_sum = vec(0, 0)
    for boid in boids:
        if boid.vel.length() == 0:
            velocity_sum += boid.vel
        else:
            velocity_sum += boid.vel.normalize()
    return velocity_sum


def draw_arrow(surface, color, start_pos, end_pos, arrow_length=5, arrow_angle=np.pi / 6):
    pygame.draw.line(surface, color, start_pos, end_pos, 2)  # Draw the main line

    # Calculate the angle of the line
    angle = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])

    # Calculate points for arrowhead
    arrowhead1 = (end_pos[0] - arrow_length * np.cos(angle - arrow_angle), end_pos[1] - arrow_length * np.sin(angle - arrow_angle))
    arrowhead2 = (end_pos[0] - arrow_length * np.cos(angle + arrow_angle), end_pos[1] - arrow_length * np.sin(angle + arrow_angle))

    # Draw arrowhead
    pygame.draw.polygon(surface, color, [end_pos, arrowhead1, arrowhead2])

# Creates the boids
def create_boids(number):
    global all_sprite, boids
    #sprite groups
    all_sprite = pygame.sprite.Group()
    #OBJECTS
    boids = [Boid() for _ in range(number)]
    boids[0].image.fill(SPECIAL_BOID_COLOR)
    all_sprite.add(boids)


def run_simulation(N,beta,show_screen):
    create_boids(N)

    #game loop
    no = 10
    run = True
    #show_screen = True
    record_video = False
    last_click = pygame.time.get_ticks()

    sum_vj = vec(0,0)

    if record_video:
        # Video writer initialization
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('boids_simulation_high_beta.mp4', fourcc, fps, (int(WIDTH), int(HEIGHT)))
        # Initialize the timer
        start_time = pygame.time.get_ticks()
        duration = 10 * 1000  # 10 seconds in milliseconds

    start_time = pygame.time.get_ticks()
    duration = 10 * 2000  # 10 seconds in milliseconds

    count = 0
    while run:
        if count == 0:
            V = compute_order_parameter(boids)
            print(np.sqrt(V[0]**2 + V[1]**2)/N)
        #clock spped
        DELTA_TIME = clock.tick(fps)/1000

        if show_screen:
            a,b,c = pygame.mouse.get_pressed()
            if a:
                if pygame.time.get_ticks() - last_click >= 400:
                    last_click = pygame.time.get_ticks()
                    create_boids(10 + no)
                    no += 10

            #input(events)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

        #update
        all_sprite.update(DELTA_TIME,beta)

        if show_screen:
            #Draw/render
            screen.fill(black)
            all_sprite.draw(screen)

            # Draw circles around each boid
            for boid in boids:
                pygame.draw.circle(screen, (255, 0, 0), (int(boid.pos.x), int(boid.pos.y)), R1, 1)  # Red color for R1
                pygame.draw.circle(screen, (255, 255, 255), (int(boid.pos.x), int(boid.pos.y)), R2, 1)  # White color for R2
                pygame.draw.circle(screen, (0, 255, 0), (int(boid.pos.x), int(boid.pos.y)), R3, 1)  # Green color for R3
                
                # Draw arrow for boid.vel 
                draw_arrow(screen, (255, 255, 255), (int(boid.pos.x), int(boid.pos.y)), (int(boid.pos.x + boid.vel.x*0.5), int(boid.pos.y + boid.vel.y*0.5)))

            if no <= 20:
                draw_text(screen, "Left click to increase Boids",25, WIDTH/2, 20)
            pygame.display.flip()

        if record_video:
            # Capture the screen and convert it to an OpenCV image
            frame = pygame.surfarray.array3d(screen)
            frame = cv2.transpose(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
            out.write(frame)
            elapsed_time = pygame.time.get_ticks() - start_time
            if elapsed_time >= duration:
                break
        
        count += 1
        elapsed_time = pygame.time.get_ticks() - start_time
        sum_vj += compute_order_parameter(boids)
        abs_sum_vj_N = np.sqrt(sum_vj[0]**2 + sum_vj[1]**2) / N
        if elapsed_time >= duration:
            break

    #pygame.quit()
    if record_video:
        out.release()

    return abs_sum_vj_N / count

def simulate_density():
    beta = 0.1
    density_values = range(20, 200, 20)     # range for density
    V_t_density = []
    for N in density_values:
        sum_vj = run_simulation(N,beta,False)
        V_t_density.append(sum_vj)
        print(N, sum_vj)

    plt.figure(1)
    plt.plot(density_values,V_t_density)
    plt.xlabel('Number of boids N')
    plt.ylabel('V(t)')
    plt.show()

def simulate_beta():
    N=30
    beta_values = np.linspace(0.1, 0.9, 10)  # range for beta
    V_t_beta = []
    for beta in beta_values:
        sum_vj = run_simulation(N,beta,False)
        V_t_beta.append(sum_vj)
        print(beta,sum_vj)

    plt.figure(1)
    plt.plot(beta_values,V_t_beta)
    plt.xlabel('beta')
    plt.ylabel('V(t)')
    plt.show()

def main():
    '''Choose one of the options,
    Set the parameters in the update function expept N and beta which is set here for the first case.
    Set the ranges R1,R2,R3 at the top'''
    N = 30
    beta = 0.1
    V = run_simulation(N,beta,True)    
    #simulate_beta()              
    #simulate_density()             
if __name__ == "__main__":
    main()