#!/usr/bin/env python3
import numpy as np
import os
import sys
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Bool
from datetime import datetime
import csv
import rospkg
from geometry_msgs.msg import PointStamped
import random
from scipy.interpolate import griddata

from functions_for_controllers import set_up_topology, evaluate_Fx_2
from scipy import optimize

class follower_longitudinal_controller_class:
	def __init__(self, car_number):

		#set up variables
		self.car_number = car_number
		self.leader_number = 0 #set_up_topology(car_number)

		self.safety_value = 0.0
		# define vehicle's mass 
		self.m = 1.67 #[Kg]

		# generate parameters
		self.v_ref = 0.60  
		self.dt = 0.1  # so first number is the prediction horizon in seconds -this is the dt of the solver so it will think that the control inputs are changed every dt seconds
		
		# 0.5 m distance high gains
		# self.k = 8.714676198486083
		# self.h = 0.3854210305823209
		# self.c = 12.200546677880515

		# 0.5 m distance in the middle of the graph
		# self.k = 4.631725987595675
		# self.h = 0.3005865102639296
		# self.c = 6.4844163826339445

		# 0.5 m distance lowest gains
		self.k = 3.453287197231834
		self.h = 0.21042084168336672
		self.c = 4.834602076124567


		self.d_safety = 0.5
		self.acc_sat = 1

		# mpc action params
		self.acc_leader_past = [0.0, 0.0]
		self.add_mpc = False
		self.b_mpc = -self.c/self.k
		self.a_mpc = self.d_safety * 0
		
		# initialize attack
		self.attack_time = 0
		self.attack_duration_threshold = 5

		# initialize state variables
		# [v v_rel x_rel]
		#self.state = [0, 0, 0]
		#self.t_prev = 0.0
		#self.t_prev_encoder = 0.0
		self.relative_state = [0.0,0.0]
		self.v = 0.0

		# initialize alarm signal
		self.alarm_threshold = 0.5
		self.attack_detected = False
		self.v_rel_estimate = 0.0
		self.K = 0.05  # equilibrium gain for the kalman filter
		self.v_prev = 0.0
		self.residual = 0.0

		#
		self.acc = 0.0

		self.attack_detection_enabled = False 



		# controller parameters subscribers
		self.safety_value_subscriber = rospy.Subscriber('safety_value', Float32, self.safety_value_subscriber_callback)
		self.safety_value_subscriber = rospy.Subscriber('v_ref', Float32, self.v_ref_callback)
		self.add_mpc_subscriber = rospy.Subscriber('add_mpc_gamepad', Bool, self.add_mpc_callback)
		#self.attack_subscriber = rospy.Subscriber('attack', Bool, self.attack_callback)
		self.sensors_and_input_subscriber = rospy.Subscriber('sensors_and_input_' + str(car_number), Float32MultiArray, self.sensors_and_input_callback)
		self.acc_publisher = rospy.Publisher('desired_acc_' + str(car_number), Float32, queue_size=1)
		self.acc_measured_publisher = rospy.Publisher('measured_acc_' + str(car_number), Float32, queue_size=1)
		
		self.relative_state_subscriber = rospy.Subscriber('relative_state_' + str(car_number), Float32MultiArray, self.relative_state_callback)
		self.alarm_publisher = rospy.Publisher('alarm_' + str(car_number), Float32, queue_size=1)
		#self.acc_leader_subscriber = rospy.Subscriber('acceleration_' + str(self.leader_number), Float32, self.acc_leader_callback)
		# state subscribers
		# set up subscribers to leader states
		self.topology_subscriber = rospy.Subscriber('topology',Float32MultiArray,self.topology_callback)
		self.attack_detection_subscriber =  rospy.Subscriber('attack_detection',Bool,self.atck_detect_callback)
		self.leader_acc_subscriber = rospy.Subscriber('u_com_' + str(self.leader_number), Float32, self.leader_acc_callback)



	def topology_callback(self,msg):
		self.leader_number =  int(msg.data[int(self.car_number)-1])
		self.leader_acc_subscriber = rospy.Subscriber('u_com_' + str(self.leader_number), Float32, self.leader_acc_callback)


	def v_ref_callback(self,msg):
		if self.leader_number == 0:
			self.v_ref = msg.data
		else:
			self.v_ref = 0.60

	# controller parameter callbacks
	def safety_value_subscriber_callback(self, msg):
		#print(msg.data)
		self.safety_value = msg.data

	def v_ref_callback(self,v_ref_msg):
		self.v_ref = v_ref_msg.data

	def add_mpc_callback(self,add_mpc_msg):
		self.add_mpc = add_mpc_msg.data



	def atck_detect_callback(self,msg):
		self.attack_detection_enabled = msg.data

	# state callbacks
	def sensors_and_input_callback(self, msg):
		self.sensors_and_input = np.array(msg.data)
		self.v_prev = self.v
		self.v = self.sensors_and_input[6]
		self.acc = (self.v-self.v_prev)/self.dt
		self.acc_measured_publisher.publish(self.acc )

	def relative_state_callback(self, msg):
		# [rel_vel, distance]
		self.relative_state = np.array(msg.data)

	# leader communication callback
	def leader_acc_callback(self,msg):
		self.acc_leader_past = [*self.acc_leader_past[1:], msg.data]


	def compute_longitudinal_control_action(self):
		if self.leader_number == 0:
			u_control = - self.h * self.k *(self.v - self.v_ref) * 5
			# artificial saturation bounds
			u_control = self.saturate_acc(u_control)
			self.alarm_publisher.publish(0.0)
			print('u control =', np.round(u_control, decimals=2))



		else:


			# introducing delay compensation by assuming the relative velocity will remain the same for a certain amount of time
			# so we are correcting everything except that we consider that the leading vehicle velocity will not change
			delay = 0.2 #[s]
			rel_state_loc = [self.relative_state[0] + delay * self.acc , self.relative_state[1] + delay * self.relative_state[0]]
			v_delay_comp = self.v + delay * self.acc


			# evaluate linear controller action
			# [rel_vel, distance]
			u_lin =  - self.k * (rel_state_loc[1]+self.d_safety) - self.h * self.k *(v_delay_comp - self.v_ref) - self.c * rel_state_loc[0]
			
			# if self.attack_detected == False:
			# 	u_ff = self.generete_mpc_action(u_lin)
			# else:
			# 	u_ff = 0.0

			if self.attack_detected and self.attack_detection_enabled:
				u_ff = 0.0
			else:
				u_ff = self.generete_mpc_action(u_lin)
				




			u_control = u_lin + u_ff

			# artificial saturation bounds
			u_control = self.saturate_acc(u_control)


			#print('u lin =', np.round(u_lin, decimals=2), ' u_ff =',np.round(u_ff, decimals=2),'sum =', np.round(u_control, decimals=2), 'v =', np.round(self.v, decimals=2))

			# evalaute alarm signal
			if self.safety_value == 1 and self.attack_detection_enabled:
				# Prediction step
				
				x_hat_minus = self.v_rel_estimate + self.dt * (self.acc - self.acc_leader_past[-1])

				# Update step
				self.v_rel_estimate = x_hat_minus * (1-self.K) + self.K * rel_state_loc[0]

				#evaluate residual
				self.residual = np.abs(rel_state_loc[0]-self.v_rel_estimate)

				#publish it
				self.alarm_publisher.publish(self.residual)

				# set ataack detection flag
				if self.residual > self.alarm_threshold:
					self.attack_time = self.attack_time + 1
					if self.attack_time > self.attack_duration_threshold:
						print('Warning anomaly detected!')
						self.attack_detected = True
				else:
					self.attack_time = 0
			else:
				#reset alarm value is safety is disingaged
				self.v_rel_estimate = 0.0



		#publish control action
		self.acc_publisher.publish(u_control)



	def generete_mpc_action(self, u_linear):
		if self.add_mpc:

			# evaluate new relative state using leader acceleration info
			# x_dot_rel_k_plus_1 = self.relative_state[0] + u_linear * self.dt -  self.acc_leader * self.dt 
			# x_rel_k_plus_1 = (self.relative_state[1]+self.d_safety) + self.relative_state[0]*self.dt

			# # max u mpc action
			u_mpc_max = self.k * (self.d_safety + self.h*(self.v -self.v_ref))
			max_x_rel_k = self.d_safety - self.c/self.k*(self.relative_state[0])

			# mpc action candidate
			#u_mpc_candidate = ((x_rel_k_plus_1 - self.a_mpc)/self.b_mpc - x_dot_rel_k_plus_1)/(3 * self.dt)
			u_mpc_candidate = self.acc_leader_past[-1]

			if (self.relative_state[1]+self.d_safety) >= max_x_rel_k:
				u_mpc = 0
				print('setting mpc to 0 because out of safety region')
			elif u_mpc_candidate > u_mpc_max:
				u_mpc = u_mpc_max
				print('setting mpc to max value')
			else:
				u_mpc = u_mpc_candidate
		
		else:
			u_mpc = 0.0

		
		return u_mpc
		
	def saturate_acc(self,acc):
		if acc >= self.acc_sat:
			acc = self.acc_sat
		elif acc <= -self.acc_sat:
			acc = -self.acc_sat

		return acc
	
	



if __name__ == '__main__':
	try:
		car_number = os.environ["car_number"]
		rospy.init_node('longitudinal_control_node_' + str(car_number), anonymous=False)
		rate = rospy.Rate(10) #Hz

		#set up longitudinal controller
		vehicle_controller = follower_longitudinal_controller_class(car_number)

		while not rospy.is_shutdown():
			#run longitudinal controller loop
			vehicle_controller.compute_longitudinal_control_action()
			rate.sleep()



	except rospy.ROSInterruptException:
		pass

