#!/usr/bin/env python

import numpy as np
import os
import sys
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Bool
import rospkg
from geometry_msgs.msg import PointStamped
from tf.transformations import euler_from_quaternion
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz,produce_marker_rviz,set_up_topology
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
import copy







class relative_state_publisher:
	# this class sets up a node that publishes the relative position and velocity between 2 vehicles. This is needed to run the longitudinal controller.
	def __init__(self, car_number):
		

		#set up variables
		self.car_number = car_number
		self.leader_number = 0 # set_up_topology(car_number)
		self.centroids = np.array([])

		self.leader_marker_array = MarkerArray()
		self.leader_marker_array.markers.append(Marker())

		# initialize state variables
		# [x y theta]
		self.state = [0, 0, 0]
		self.leader_position = [0, 0]
		self.previous_path_index = 0 # initial index for closest point in global path
		self.sensors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		self.v = 0.1
		self.v_leader = 0.0

		# initialize filter variables
		f_cutoff = 10.0  # cutoff frequency in Hz
		dt = 0.1 # [s]
		# Calculate the filter coefficient 'a' using the time constant
		tau = 1 / (2 * np.pi * f_cutoff)
		self.a = 1.0 #dt / (tau + dt)   # setting to 1 means no filter
		self.leader_to_follower_distance_prev = 0.0
		self.vel_rel_prev = 0.0

		# set maximum distance between global leader position and lidar centroid position
		self.max_dist_from_leader_initial_guess = 0.6

		# Set DBSCAN parameters
		self.lidar_centroid_markers = MarkerArray() 

		# set up publisher
		self.relative_state_publisher = rospy.Publisher('relative_state_' + str(self.car_number), Float32MultiArray, queue_size=10)

		#subscribers
		#self.rviz_closest_point_on_path_leader = rospy.Subscriber('rviz_closest_point_on_path_' + str(self.leader_number), Marker, self.leader_path_progress_callback)
		self.state_subscriber = rospy.Subscriber('sensors_and_input_' + str(self.car_number), Float32MultiArray, self.sensors_callback)
		self.leader_state_subscriber = rospy.Subscriber('sensors_and_input_' + str(self.leader_number), Float32MultiArray, self.sensors_leader_callback)
		rospy.Subscriber('/clusters_' + str(car_number), MarkerArray, self.clusters_callback)
		self.leader_marker_publisher = rospy.Publisher('leader_position_' + str(self.car_number), Marker, queue_size=10)
		self.tf_listener = tf.TransformListener()

		self.topology_subscriber = rospy.Subscriber('topology',Float32MultiArray,self.topology_callback)


	def topology_callback(self,msg):
		
		self.leader_number =  int(msg.data[int(self.car_number)-1])
		self.leader_state_subscriber = rospy.Subscriber('sensors_and_input_' + str(self.leader_number), Float32MultiArray, self.sensors_leader_callback)

	def sensors_callback(self, msg):
		sensors = np.array(msg.data)
		#elapsed_time, current,voltage,IMU[0](acc x),IMU[1] (acc y),IMU[2] (omega rads),velocity, safety, throttle, steering
		self.v = sensors[6]

	def sensors_leader_callback(self, msg):
		sensors = np.array(msg.data)
		#elapsed_time, current,voltage,IMU[0](acc x),IMU[1] (acc y),IMU[2] (omega rads),velocity, safety, throttle, steering
		self.v_leader = sensors[6]

	def leader_path_progress_callback(self, msg):
		self.leader_position = [msg.pose.position.x,msg.pose.position.y]

	def evaluate_centroids(self,marker_array):
		cluster_centroids = []

		for marker in marker_array.markers:
			if marker.type == Marker.POINTS:
				# Extract cluster points from Marker
				cluster_points = np.array([(point.x, point.y) for point in marker.points])

				# Calculate the centroid of the cluster
				if len(cluster_points) > 0:
					centroid = np.mean(cluster_points, axis=0)
					cluster_centroids.append(centroid)

		return np.array(cluster_centroids)

	def clusters_callback(self, marker_array):
		# Evaluate centroids
		self.centroids = self.evaluate_centroids(marker_array)




	def create_leader_marker(self, x, y, frame_id):
		# Create a PointStamped message to represent the position in the desired frame
		point_stamped = PointStamped()
		point_stamped.header.frame_id = frame_id
		point_stamped.point.x = x
		point_stamped.point.y = y

		# Create a Marker message
		marker = Marker()
		marker.header = point_stamped.header
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = point_stamped.point.x
		marker.pose.position.y = point_stamped.point.y
		marker.pose.position.z = 0.0  # Assuming z = 0 in a 2D space
		marker.scale.x = marker.scale.y = marker.scale.z = 0.1  # Adjust the scale as needed
		marker.color.r = 0.8 
		marker.color.g = 0.4 
		marker.color.b = 0.0
		marker.color.a = 1.0  # Alpha

		return marker


	def evaluate_relative_state(self):
		if self.leader_number == 0: # this vehicle is the leader so no need to evaluate relative state
			# publish zeros
			floatarray_msg = Float32MultiArray()
			floatarray_msg.data = [0.0, 0.0]
			self.relative_state_publisher.publish(floatarray_msg)
		else:
			#get latest transform data for robot pose
			#self.tf_listener.waitForTransform("/map", "/base_link_" + str(self.car_number), rospy.Time(), rospy.Duration(1.0))
			#(robot_position,robot_quaternion) = self.tf_listener.lookupTransform("/map",  "/base_link_" + str(self.car_number), rospy.Time(0))

			#get latest transform data for leader
			self.tf_listener.waitForTransform("/scan_" + str(self.car_number), "/base_link_" + str(self.leader_number), rospy.Time(), rospy.Duration(1.0))
			(leader_position,leader_quaternion) = self.tf_listener.lookupTransform("/scan_" + str(self.car_number),  "/base_link_" + str(self.leader_number), rospy.Time(0))


			## JUST FOR TESTINGGGG !!!!!!!!!!
			#leader_position = np.array([0.5, 0])


			# copy global variable to avoit it being updated during this loop
			centroids_local = copy.copy(self.centroids)
			# find closest lidar centroid marker to the leader's position
			closest_centroid_distance = float('inf')
			closest_index = None
			for i in range(centroids_local.shape[0]):
				# Calculate Euclidean distance
				distance = np.sqrt((leader_position[0] - centroids_local[i,0])**2 + (leader_position[1] - centroids_local[i,1])**2)

				# Update closest marker if the current one is closer
				if distance < closest_centroid_distance and distance < self.max_dist_from_leader_initial_guess:
					closest_centroid_distance = distance
					closest_index = i


			if closest_index is not None:
				# update leader position using the one measured from the lidar
				leader_position = centroids_local[closest_index,:]
				
			else:
				#print('no cluster found close to leader global position, using the latter instead')
				# overwrite distance if no reasonable guess was found
				pass

			leader_to_follower_distance_current = np.sqrt((leader_position[0])**2 + (leader_position[1])**2) # the leader position is already in the follower's frame of reference
			leader_marker = self.create_leader_marker(leader_position[0],leader_position[1],'scan_' + str(self.car_number))

			self.leader_marker_publisher.publish(leader_marker)
				

			#evalaute relative velocity betwwen ego vehicle and leader
			rel_vel_current = self.v-self.v_leader

			# filtering the signal
			rel_vel = (1 - self.a) * self.vel_rel_prev + self.a * rel_vel_current
			leader_to_follower_distance = (1 - self.a) * self.leader_to_follower_distance_prev + self.a * leader_to_follower_distance_current
			# acoount for extra length
			leader_to_follower_distance = leader_to_follower_distance #- 0.087
			#update previous quantities
			self.vel_rel_prev = rel_vel
			self.leader_to_follower_distance_prev = leader_to_follower_distance

			#publish relative state
			floatarray_msg = Float32MultiArray()

			if leader_position[0] > 0:
				sign = -1
			else:
				sign = 1


			floatarray_msg.data = [rel_vel, sign * leader_to_follower_distance +0.08]
			self.relative_state_publisher.publish(floatarray_msg)
		




if __name__ == '__main__':
	try:
		car_number = os.environ["car_number"]
		rospy.init_node('relative_state_publisher_node_' + str(car_number), anonymous=False)
		rate = rospy.Rate(10) #Hz

		#set up relative distance evaluator. Note that is measures the distance between the projections on the global path
		relative_state_publisher_obj = relative_state_publisher(car_number)


		while not rospy.is_shutdown():
			#run evalaution in a loop
			try:
				relative_state_publisher_obj.evaluate_relative_state()
			except Exception as error:
				#pass
				print('failed to evaluate relative position:', error)




	except rospy.ROSInterruptException:
		pass
