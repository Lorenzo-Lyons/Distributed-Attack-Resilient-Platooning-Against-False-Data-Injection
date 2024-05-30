#!/usr/bin/env python3

import rospy
import pygame
from std_msgs.msg import Float32, Bool, Float32MultiArray
import os
import numpy as np









# Initialize pygame
pygame.init()
# j = pygame.joystick.Joystick(0)
# j.init()
# print ('Initialized Joystick : %s' % j.get_name())

# Set display dimensions
width, height = 1350, 420
gameDisplay = pygame.display.set_mode((width, height))
pygame.display.set_caption("Teleop controller")

# Set up Pygame font
font = pygame.font.Font(None, 36)  # Use the default font with size 36

# Set font color (RGB tuple)
font_color = (50, 50, 50)  # White

# define colors
color_hex_active = "ABE9FC"
active_color = tuple(int(color_hex_active[i:i+2], 16) for i in (0, 2, 4))


# Define colors
#active_color = (0, 255, 0)  # Green when activated
inactive_color = (224, 224, 224)  # Red when deactivated


# Load the image
image_base = pygame.image.load("platooning_ws/src/platooning_utilities/scripts/images/base.png")
image_comm = pygame.image.load("platooning_ws/src/platooning_utilities/scripts/images/comm.png")
image_atck = pygame.image.load("platooning_ws/src/platooning_utilities/scripts/images/atck.png")
image_atck_detection = pygame.image.load("platooning_ws/src/platooning_utilities/scripts/images/atck_detection.png")
image_atck_detected = pygame.image.load("platooning_ws/src/platooning_utilities/scripts/images/atck_detected.png")
image_atck_isolated = pygame.image.load("platooning_ws/src/platooning_utilities/scripts/images/atck_isolated.png")
image_topology_1 = pygame.image.load("platooning_ws/src/platooning_utilities/scripts/images/topology_1.png")
image_topology_2 = pygame.image.load("platooning_ws/src/platooning_utilities/scripts/images/topology_2.png")

def draw_entry(text, key, x, y, active, entry_width, entry_height):
    
    # Define vertical spacing between entries
    vertical_spacing = 10
    text_right_shift = 15
    y_up_shift_from_bottom = 10
    
    # Draw filled rectangle around the entry with the specified color
    entry_rect = pygame.Rect(x, y, entry_width, entry_height)
    pygame.draw.rect(gameDisplay, active_color if active else inactive_color, entry_rect,  border_radius=15)
    
    # Draw border around the filled rectangle
    border_color = (0,0,0)
    pygame.draw.rect(gameDisplay, border_color, entry_rect, 2,  border_radius=15)

    # Render and display the first line of text
    text_surface = font.render(text, True, font_color)
    gameDisplay.blit(text_surface, (x + text_right_shift, y + y_up_shift_from_bottom))

    # Render and display the second line of text (key)
    smaller_font = pygame.font.Font(None, 20)  # Adjust font size as needed
    key_surface = smaller_font.render(key, True, font_color)
    gameDisplay.blit(key_surface, (x + entry_width + 10, y + y_up_shift_from_bottom))  # Adjust the y-position as needed

    
    # Increase y-coordinate for next entry
    return y + entry_height + vertical_spacing






class GUI_class:
    def __init__(self):
        

        self.add_ff_gamepad = False
        self.overtaking = False
        self.platoon_rearranged =False
        self.attack = False
        self.enable_attack_detection = False
        self.emergency_brake = False
        self.safety_value = False
        self.attack_detected = False
        self.alarm_counter = 0

        self.topology = 1 # can be 1 or 2

        # Initialize node
        rospy.init_node('demo_subscriber', anonymous=True)

        # Setup subscribers
        rospy.Subscriber('add_mpc_gamepad', Bool, self.add_ff_callback)
        rospy.Subscriber('topology', Float32MultiArray, self.topology_callback)
        #rospy.Subscriber('overtaking_1', Bool, self.overtaking_1_callback)
        rospy.Subscriber('overtaking_2', Bool, self.overtaking_2_callback)
        #rospy.Subscriber('overtaking_3', Bool, self.overtaking_3_callback)
        #rospy.Subscriber('overtaking_4', Bool, self.overtaking_4_callback)
        rospy.Subscriber('attack', Bool, self.attack_callback)
        rospy.Subscriber('attack_detection', Bool, self.attack_detection_callback)
        rospy.Subscriber('v_ref', Float32, self.v_ref_callback)
        rospy.Subscriber('safety_value', Float32, self.safety_callback)
        rospy.Subscriber('alarm_2', Float32, self.alarm_2_callback)



        rate = rospy.Rate(10)  # 10hz

        # Get start time
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.overtaking:
                self.platoon_rearranged = True

            pygame.event.pump()

            # Clear the screen
            gameDisplay.fill((255, 255, 255))

            # Display gamepad settings
            entries = [
                ("Sensor-based control", "(R1)", True),
                ("Communication", "(L1)", self.add_ff_gamepad),
                ("Attack", "(L2)", self.attack),
                ("Attack detection", "(sqr)", self.enable_attack_detection),
                ("Topology changed", "(R2)", self.topology==2),
                ("Emergency brake", "(X-tri)", self.emergency_brake)
            ]

            # Define box dimensions
            entry_width = 285
            entry_height = 47 #76  # Increased height

            y = 50
            x = 50
            for i, (text, key, active) in enumerate(entries):
                y = draw_entry(text, key, x, y, active, entry_width, entry_height)

            # Blit the image onto the display surface
            y_image = 50
            x_image = x + entry_width + 50
            gameDisplay.blit(image_base, (x_image, y_image))

            print(self.attack_detected)
            # quick fix on attack detected
            if self.enable_attack_detection == False:
                self.attack_detected = False

            # add other images if neede
            if self.add_ff_gamepad and not self.attack_detected :
                gameDisplay.blit(image_comm, (x_image, y_image))

            if self.attack and self.topology==1 and not self.attack_detected :
                gameDisplay.blit(image_atck, (x_image, y_image))

            if self.enable_attack_detection:
                gameDisplay.blit(image_atck_detection, (x_image, y_image))

            if self.attack_detected and self.topology == 1:
                gameDisplay.blit(image_atck_detected, (x_image, y_image))

            if self.topology == 2 and self.attack:
                gameDisplay.blit(image_atck_isolated, (x_image, y_image))

            if self.topology == 1:
                gameDisplay.blit(image_topology_1, (x_image, y_image))
            else:
                gameDisplay.blit(image_topology_2, (x_image, y_image))

            # Update the display
            pygame.display.flip()

            rate.sleep()



    def add_ff_callback(self, msg):
        self.add_ff_gamepad = msg.data

    def overtaking_2_callback(self, msg):
        self.overtaking = msg.data


    def attack_callback(self, msg):
        self.attack = msg.data

    def attack_detection_callback(self, msg):
        self.enable_attack_detection = msg.data

    def topology_callback(self, msg):
        if msg.data[0] == 0:
            self.topology = 1
        else:
            self.topology = 2

    def v_ref_callback(self, msg):
        if msg.data == 0.0:
            self.emergency_brake = True
        else:
            self.emergency_brake = False

    def safety_callback(self, msg):
            self.safety_value = msg.data

    def alarm_2_callback(self, msg):
        if self.topology == 1 and self.enable_attack_detection:
            if msg.data > 0.5:
                if self.alarm_counter > 5 :
                    self.attack_detected = True
                else:
                    self.alarm_counter = self.alarm_counter + 1
            else:
                self.alarm_counter = 0
        else:
            self.attack_detected = False



if __name__ == '__main__':
    try:
        GUI_class()
    except rospy.ROSInterruptException:
        pass












