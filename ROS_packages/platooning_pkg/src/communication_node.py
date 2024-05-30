#!/usr/bin/env python3
import numpy as np
import os
import rospy
from std_msgs.msg import Float32, Bool
import random

class comm_class:
	def __init__(self, car_number):

		#set up variables
		self.car_number = car_number

		# initialize state variables
		self.safety_value = 0

		#initialize variables
		self.attack = False
		self.safety_value = 0 


		# set up subscribers
		self.attack_subscriber = rospy.Subscriber('attack', Bool,self.attack_callback)
		self.safety_subscriber = rospy.Subscriber('safety_value', Float32,self.safety_value_subscriber_callback)
		self.acc_saturated_subscriber = rospy.Subscriber('acc_saturated_' + str(car_number), Float32,self.acc_callback)
		#self.acc_measured_subscriber = rospy.Subscriber('measured_acc_' + str(car_number), Float32,self.acc_callback)


		# acc comm publisher
		self.acc_comm_publisher = rospy.Publisher('u_com_' + str(car_number), Float32, queue_size=1)

		# time in case outputting a sinusoidal acc instead for system validation
		self.start_time = rospy.Time.now()

	# controller parameter callbacks

	def safety_value_subscriber_callback(self, msg):
		#print(msg.data)
		self.safety_value = msg.data

	def attack_callback(self, msg):
		#print(msg.data)
		self.attack = msg.data


	# state callbacks
	def acc_callback(self, msg):
		if self.safety_value == 1:
			if self.attack and int(self.car_number)==1: # define attack logic here
				current_time = rospy.Time.now()
				elapsed_time = current_time.to_sec() - self.start_time.to_sec()
				amp = 1
				freq = 0.05
				communicated_acc = amp * np.sin(freq * elapsed_time * 2 * np.pi)
				if communicated_acc > 0:
					communicated_acc = 1
				else:
					communicated_acc = -1

				# random noise
				#communicated_acc = msg.data + random.gauss(0, 0.5)
			else:
				communicated_acc = msg.data

		else:
			communicated_acc = 0.0

		self.acc_comm_publisher.publish(communicated_acc)




if __name__ == '__main__':
	try:
		car_number = os.environ["car_number"]
		rospy.init_node('communication_node_' + str(car_number), anonymous=False)

		#set up longitudinal controller
		comm_class_obj = comm_class(car_number)

		# spin to keep node going
		rospy.spin()

	except rospy.ROSInterruptException:
		pass

