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
from scipy import optimize

class acc_2_throttle_class:
	def __init__(self, car_number):

		#set up variables
		self.car_number = car_number
		#self.leader_number = set_up_topology(car_number)

		# define vehicle's mass 
		self.m = 1.67 #[Kg]

		# initialize state variables
		self.actuation_delay_steps = 2 #[s]
		self.dt = 0.1 # NOTE this must match the actual controller frequency
		self.v = 0.0
		self.v_past = 0.0
		self.past_desired_acc = [0,0,0,0,0 ,0,0,0,0,0]
		self.past_acc = [0,0,0,0,0 ,0,0,0,0,0]
		# this is needed to close the loop on acceleration
		self.acc_feedback_gain_int = 0.001
		self.tau_fb_max_int = 0.1
		self.tau_fb_min_int = -0.1
		self.acc_feedback_gain_proportional = 0.00 #0.05 #0.25
		self.tau_fb_max = 0.1
		self.tau_fb_min = -0.1
		self.v_desired_future = 0.0
		self.tau_fb_int_default = 0.04
		self.tau_fb_int = self.tau_fb_int_default

		self.safety_value = 0

		# define grid data to avoid inverting the model dynamics
		# Create a grid of throttle and velocity values
		# set throttle limits
		self.min_tau = 0.0
		if float(car_number) == 2:
			self.max_tau = 0.35 #0.35
		else:
			self.max_tau = 0.30 #0.35


		self.throttle_values = np.linspace(self.min_tau , self.max_tau , 20)  # Adjust the range and resolution as needed
		self.acc_vals = self.acceleration(self.throttle_values, self.v)

		# set up publisher and subscribers
		self.throttle_publisher = rospy.Publisher('throttle_' + str(car_number), Float32, queue_size=1)
		self.acc_saturated_publisher = rospy.Publisher('acc_saturated_' + str(car_number), Float32, queue_size=1)

		# controller parameters subscribers
		self.safety_value_subscriber = rospy.Subscriber('safety_value', Float32, self.safety_value_subscriber_callback)
		self.desired_acc_subscriber = rospy.Subscriber('desired_acc_' + str(car_number), Float32, self.desired_acc_callback)

		# state subscribers
		self.sensors_and_input_subscriber = rospy.Subscriber('sensors_and_input_' + str(car_number), Float32MultiArray, self.sensors_and_input_callback)
		
		# time in case outputting a sinusoidal acc instead for system validation
		self.start_time = rospy.Time.now()

	# controller parameter callbacks

	def safety_value_subscriber_callback(self, msg):
		#print(msg.data)
		self.safety_value = msg.data

	# state callbacks

	def sensors_and_input_callback(self, msg):
		self.sensors_and_input = np.array(msg.data)

		self.v_past = self.v
		self.v = self.sensors_and_input[6]   #np.mean([self.sensors_and_input[6], *self.v_past])

		self.past_acc = [*self.past_acc[1:],(self.v-self.v_past)/self.dt]
		
		
		# update acc value according to current velocity
		self.acc_vals = self.acceleration(self.throttle_values, self.v)


	# utility

	def publish_throttle(self, tau):
		# saturation limits for tau
		if tau < 0:
			tau = 0
		elif tau > self.max_tau:
			tau = self.max_tau

		throttle_val = Float32(tau)

		#publish inputs
		self.throttle_publisher.publish(throttle_val) 

	# dynamic model

	def motor_force(self,th,v):
		if float(self.car_number) == 1 or float(self.car_number) == 3:
			a_m =  28.887779235839844
			b_m =  5.986172199249268
			c_m =  -0.15045104920864105

		elif float(self.car_number) == 2:
			a_m =  26.47014617919922
			b_m =  8.640666007995605
			c_m =  -0.1981888711452484
		
		elif float(self.car_number) == 4:
			# a_m =  27.996997833251953
			# b_m =  5.222377300262451
			# c_m =  -0.1444309949874878

			a_m =  28.323787689208984
			b_m =  8.21423053741455
			c_m =  -0.13714951276779175

		w = 0.5 * (np.tanh(100*(th+c_m))+1)
		Fm =  (a_m - v * b_m) * w * (th+c_m)
		return Fm

	def friction(self,v):
		if float(self.car_number) == 1 or float(self.car_number) == 3:
			a_f =  1.7194761037826538
			b_f =  13.312559127807617
			c_f =  0.289848655462265

		elif float(self.car_number) == 2:
			a_f =  1.6498010158538818
			b_f =  15.262519836425781
			c_f =  0.009999999776482582
		
		elif float(self.car_number) == 4:
			a_f =  1.767649531364441
			b_f =  13.065838813781738
			c_f =  0.009999999776482582

			
		Ff = - a_f * np.tanh(b_f  * v) - v * c_f
		return Ff

	def acceleration(self,th,v):
		return (self.motor_force(th,v) + self.friction(v))/self.m


	# deisred acc callback

	def desired_acc_callback(self, desired_acceleration_msg):

		# # bypassing desired acceleration
		# amp = 2
		# freq = 1
		# current_time = rospy.Time.now()
		# elapsed_time = current_time.to_sec() - self.start_time.to_sec()
		# u_lin = amp * np.sin(elapsed_time * freq)



		desired_acceleration = desired_acceleration_msg.data

		if desired_acceleration < np.min(self.acc_vals):
			desired_acceleration = np.min(self.acc_vals)
			#print('the requested acceleration is less than min acceleration (for the current velocity), seetting tau to 0')
		elif desired_acceleration > np.max(self.acc_vals):
			desired_acceleration = np.max(self.acc_vals)
			#print('the requested acceleration is more than max acceleration (for the current velocity), seetting tau to maximum')

		self.acc_saturated_publisher.publish(desired_acceleration)
		# update past desired accelerations
		self.past_desired_acc = [*self.past_desired_acc[1:], desired_acceleration]

		# evaluate current acceleration 
		#acc = (self.v-self.v_past)/self.dt

		#print('list list',np.array([1,2,3])-np.array([0,1,2]))
		past_delta_acc = np.array(self.past_desired_acc[:-self.actuation_delay_steps]) - np.array(self.past_acc[self.actuation_delay_steps:])


		# integral action correction term
		if self.safety_value == 1: # update the correction term only is safety is off
			#self.tau_fb_int = self.tau_fb_int + self.acc_feedback_gain_int * (self.v_desired_future - self.v)
			self.tau_fb_int = self.tau_fb_int + self.acc_feedback_gain_int * (np.mean(past_delta_acc))
			self.tau_fb_int = np.min([self.tau_fb_int, self.tau_fb_max_int])
			self.tau_fb_int = np.max([self.tau_fb_int, self.tau_fb_min_int])
		else:
			self.tau_fb_int = self.tau_fb_int_default
			

		self.tau_fb = self.tau_fb_int + self.acc_feedback_gain_proportional * (np.mean(past_delta_acc))
		#print('tau_fb int', self.tau_fb_int, 'tau_fb prop', self.acc_feedback_gain_proportional * (np.mean(past_delta_acc)))
		#saturate feedback action
		self.tau_fb = np.min([self.tau_fb, self.tau_fb_max])
		self.tau_fb = np.max([self.tau_fb, self.tau_fb_min])

		tau = np.interp(desired_acceleration, self.acc_vals, self.throttle_values, left=self.min_tau,right=self.max_tau) 

		# evalaute what the next velocity should be 1 control loop in the future
		self.v_desired_future = self.v + desired_acceleration * self.dt

		#print('tau =', tau, '  self.tau_fb=', self.tau_fb, '  self.tau_fb int=', self.tau_fb_int, '  self.tau_fb p=', self.acc_feedback_gain_proportional * (self.v_desired_future - self.v))
		self.publish_throttle(tau + self.tau_fb)



if __name__ == '__main__':
	try:
		car_number = os.environ["car_number"]
		rospy.init_node('acc_2_throttle_control_node_' + str(car_number), anonymous=False)

		#set up longitudinal controller
		acc_2_throttle_obj = acc_2_throttle_class(car_number)

		# spin to keep node going
		rospy.spin()




	except rospy.ROSInterruptException:
		pass

