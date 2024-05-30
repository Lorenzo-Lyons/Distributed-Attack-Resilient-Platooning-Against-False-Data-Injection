#!/usr/bin/env python3

import rospy
import pygame
import time
from std_msgs.msg import Float32,Bool, Float32MultiArray
import os
import numpy as np

# this gamepad publishes a reference velocity for the longitudinal motion.

#this allows to run the gamepad without a video display plugged in!
os.environ["SDL_VIDEODRIVER"] = "dummy"

#Initialize pygame and gamepad
pygame.init()
j = pygame.joystick.Joystick(0)
j.init()
print ('Initialized Joystick : %s' % j.get_name())
print('remove safety by pressing R1 button')

def teleop_gamepad(car_number):

	#Setup topics publishing and nodes
	pub_v_ref = rospy.Publisher('v_ref', Float32, queue_size=8)
	pub_safety_value = rospy.Publisher('safety_value', Float32, queue_size=8)
	pub_topology = rospy.Publisher('topology', Float32MultiArray, queue_size=8)
	pub_add_mpc = rospy.Publisher('add_mpc_gamepad', Bool, queue_size=8)
	pub_overtaking_1 = rospy.Publisher('overtaking_1', Bool, queue_size=8)
	pub_overtaking_2 = rospy.Publisher('overtaking_2', Bool, queue_size=8)
	pub_overtaking_3 = rospy.Publisher('overtaking_3', Bool, queue_size=8)
	pub_overtaking_4 = rospy.Publisher('overtaking_4', Bool, queue_size=8)
	pub_attack = rospy.Publisher('attack', Bool, queue_size=8)
	pub_attack_detection = rospy.Publisher('attack_detection', Bool, queue_size=8)
	pub_emergency_brake = rospy.Publisher('emergency_brake', Bool, queue_size=8)

	rospy.init_node('teleop_gamepad' + str(car_number), anonymous=True)
	rate = rospy.Rate(10) # 10hz

	# initialize v_ref
	v_ref_mean_default = 0.60 # [m/s]
	v_ref_mean = v_ref_mean_default
	incr = 0.1 # increase by this amount

	#sine signal parameters
	amp = 0.5
	freq = 0.5 # Hz
	sinusoidal_v_ref = False
	add_mpc_gamepad = False
	overtaking_1 = False
	overtaking_2 = False
	overtaking_3 = False
	overtaking_4 = False
	attack = False
	enable_attack_detection = False
	topology = np.array([0,1,2,3]) #only 2 cars
	R2_click = 1 # cycles through r2 button
	emergency_brake = False

	# get time now
	start_time = rospy.Time.now()

	while not rospy.is_shutdown():
		pygame.event.pump()

		# Collect and publish safety value
		if j.get_button(5) == 1:
			#print('safety off')
			pub_safety_value.publish(1)
		else:
			pub_safety_value.publish(0)

		current_time = rospy.Time.now()
		elapsed_time = current_time.to_sec() - start_time.to_sec()

		# increase reference velocity by pressing the "Y" and "A" buttons
	
		for event in pygame.event.get(): # User did something.
			if event.type == pygame.JOYBUTTONDOWN: 
				if j.get_button(3) == 1: # button X
					if enable_attack_detection:
						enable_attack_detection = False
					else:
						enable_attack_detection = True


				if j.get_button(1) == 1: # button B
					if sinusoidal_v_ref:
						sinusoidal_v_ref = False
					else:
						sinusoidal_v_ref = True

				if j.get_button(2) == 1: # button Y
					v_ref_mean = v_ref_mean_default 
					emergency_brake = False

				if j.get_button(0) == 1: # button A
					v_ref_mean = 0
					emergency_brake = True
					

				if j.get_button(4) == 1: # L1
					if add_mpc_gamepad:
						add_mpc_gamepad = False
					else:
						add_mpc_gamepad = True
					
				if j.get_button(6) == 1: # L2
					if attack:
						attack = False
					else:
						attack = True
					

				if j.get_button(7) == 1: # R2
					#topology = np.array([3,0,2])
					#topology = np.array([4,0,2,3])
					if R2_click == 1:
						overtaking_2 = True
						overtaking_3 = True
						overtaking_4 = True
						topology = np.array([4,0,2,3]) # with only 2 cars
					elif R2_click == 2:
						overtaking_2 = False
						overtaking_3 = False
						overtaking_4 = False
					elif R2_click == 3:
						topology = np.array([0,1,2,3]) # with only 2 cars
						overtaking_1 = True
					elif R2_click == 4:
						overtaking_1 = False

					R2_click = R2_click + 1
					if R2_click > 4:
						R2_click = 1 # reset to 1

			# print gamepad settings
			#print('sinusoidal_v_ref =',sinusoidal_v_ref,'   v_ref_mean =',v_ref_mean,
			#'   ff action =',add_mpc_gamepad,'   attack =',attack,'   attack detection=',enable_attack_detection,
			#'   overtaking =',overtaking_234)
				
		if sinusoidal_v_ref:
			pass
			#v_ref_sinusoidal =  amp * np.sin(freq * elapsed_time * 2 * np.pi)
		else:
			pass
			#v_ref_sinusoidal = 0

		v_ref = v_ref_mean #+ v_ref_sinusoidal

		# publish thigs
		pub_v_ref.publish(v_ref)
		pub_add_mpc.publish(add_mpc_gamepad)
		pub_attack.publish(attack)
		pub_attack_detection.publish(enable_attack_detection)
		pub_overtaking_1.publish(overtaking_1) 
		pub_overtaking_2.publish(overtaking_2)
		pub_overtaking_3.publish(overtaking_3)
		pub_overtaking_4.publish(overtaking_4)
		floatarray_msg = Float32MultiArray()
		floatarray_msg.data = topology
		pub_topology.publish(floatarray_msg)
		pub_emergency_brake.publish(emergency_brake)

		rate.sleep()

if __name__ == '__main__':
	try:
		try:
			car_number = os.environ['car_number']
		except:
			car_number = 1 # set to 1 if environment variable is not set
		teleop_gamepad(car_number)
	except rospy.ROSInterruptException:
		pass