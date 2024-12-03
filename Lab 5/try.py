import pygame
from sys import exit
from random import randint, uniform, choice
import numpy as np

WIDTH = 800
HEIGHT = 600
fps = 20

black = (0,0,0)
white = (255,255,255)

#for display
pygame.init()
screen = pygame.display.set_mode((WIDTH,HEIGHT))
try: pygame.display.set_icon(pygame.image.load("nboids.png"))
except: print("Note: nboids.png icon not found, skipping..")
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
BOID_SIZE = 8

R1 = 43
R2 = 120
R3 = 400

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
        self.acc = vec(0, 0)
        self.rect.center = self.pos

        self.repell = [] # Zone 1
        self.align  = [] # Zone 2
        self.attract  = [] # Zone 3

    # Uppdate function
    def update(self, DELTA_TIME):
        # Variables
        rho1 = 1  # repell factor
        rho2 = 0  # align factor
        rho3 = 0  # attract factor
        rho4 = 0  # same dir factor
        alpha = 1   # influence of speed
        beta = 0    # influence of brownian motion

        # Update influence groups 
        self.influence(boids)

        # Uppdate velocity (*MAX_SPEED otherwise it is to slow)
        self.vel = (rho1*self.repell_vec() + rho2*self.align_vec() + rho3*self.attract_vec() + rho4*self.vel) * MAX_SPEED

        # Ensures that all speeds are constants
        # not sure why this was needed since all vectors are normalized and rhos add up to 1 but through experiments i needed it
        if self.vel.length() > MAX_SPEED:
            self.vel.scale_to_length(MAX_SPEED)

        # Update position with brownian motion
        self.pos += alpha*self.vel*DELTA_TIME + beta*self.brownian_motion(DELTA_TIME)

        # Create the boundary condition
        if self.pos.x > WIDTH:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.y > HEIGHT:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = HEIGHT

        self.rect.center = self.pos

    # Brownian motion function
    def brownian_motion(self, delta_t):
        c = 3
        r_max = 10  # Choose your maximum value of |X(∆t)| here
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
        for b in boids:
            if b != self:
                distance = self.wrap_distance(self.pos, b.pos)
                if distance.length() < R1:
                    self.repell += [b]
                elif distance.length() < R2:
                    self.align += [b]
                elif distance.length() < R3:
                    self.attract += [b]

    # Function to take into account the borders returns min distance between boids
    def wrap_distance(self, pos1, pos2):
        dx = min(abs(pos1.x - pos2.x), WIDTH - abs(pos1.x - pos2.x))
        dy = min(abs(pos1.y - pos2.y), HEIGHT - abs(pos1.y - pos2.y))
        return vec(dx, dy)

    # Function to calculate the center of mass for a group
    def center_of_mass(self,part):
        center = vec(0,0)
        if len(part) == 0:
            return center
        for b in part:
            wrapped_pos = self.pos + self.wrap_distance(self.pos, b.pos)
            center += wrapped_pos
        return center / len(part)
    

    # Function for separation (repelling)
    def repell_vec(self):
        center = self.center_of_mass(self.repell)
        e_1 = self.pos - center
        if e_1.length() == 0:
            return e_1
        else:
            return e_1.normalize()

     # Function for allignment use average v_2(t)
    def align_vec(self):
        if len(self.align) == 0:
            return vec(0,0)
        e_2 = vec(0,0)
        for i in self.align:
            e_2 += i.vel
        e_2 = e_2 /len(self.align)
        if e_2.length() == 0:
            return e_2
        else:
            return e_2.normalize()

    # Function for cohesion (attraction)
    def attract_vec(self):
        e_3 = vec(0,0)
        center = self.center_of_mass(self.attract)
        e_3 = center - self.pos
        if e_3.length() == 0:
            return e_3
        else:
            return e_3.normalize()

# Creates the boids
def create_boids(number):
    global all_sprite, boids
    #sprite groups
    all_sprite = pygame.sprite.Group()
    #OBJECTS
    boids = [Boid() for _ in range(number)]
    boids[0].image.fill(SPECIAL_BOID_COLOR)
    all_sprite.add(boids)

create_boids(10)

#game loop
no = 10
run = True
last_click = pygame.time.get_ticks()
while run:
    #clock spped
    DELTA_TIME = clock.tick(fps)/1000

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
    all_sprite.update(DELTA_TIME)

    #Draw/render
    screen.fill(black)
    all_sprite.draw(screen)
    if no <= 20:
        draw_text(screen, "Left click to increase Boids",25, WIDTH/2, 20)
    pygame.display.flip()

pygame.quit()
exit()