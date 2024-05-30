#!/usr/bin/env python3

#!/usr/bin/env python3

import pygame
pygame.init()

# Initialize the first joystick
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

s = pygame.display.set_mode((640, 480))

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:  # The user closed the window!
            running = False  # Stop running

        # Logic goes here
        elif e.type == pygame.JOYBUTTONDOWN:
            button = e.button
            print(f"Button {button} is pressed.")

        elif e.type == pygame.JOYBUTTONUP:
            button = e.button
            print(f"Button {button} is released.")

        elif e.type == pygame.JOYAXISMOTION:
            axis = e.axis
            value = e.value
            print(f"Axis {axis} moved to {value}.")

pygame.quit()
