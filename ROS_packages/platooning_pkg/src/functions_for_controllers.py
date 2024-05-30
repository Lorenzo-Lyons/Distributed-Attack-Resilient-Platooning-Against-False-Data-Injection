import numpy as np
import sys
import time
import math
from visualization_msgs.msg import MarkerArray, Marker
import rospy
from geometry_msgs.msg import Point




def set_up_topology(car_number):
	if car_number == '1':
		leader_number = 0
	if car_number == '2':
		leader_number = 1
	if car_number == '3':
		leader_number = 2
	return leader_number



# Build the track structure for each available choice
def produce_track(choice,n_checkpoints):
	print('track choice = ' + choice)
	if choice == 'savoiardo':

		R = 0.8  
		theta_init2 = np.pi * -0.5
		theta_end2 = np.pi * 0.5
		theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)
		theta_init4 = np.pi * 0.5
		theta_end4 = np.pi * 1.5
		theta_vec4 = np.linspace(theta_init4, theta_end4, n_checkpoints)

		Checkpoints_x1 = np.linspace(- 1.5 * R, 1.5 * R, n_checkpoints)
		Checkpoints_y1 = np.zeros(n_checkpoints) - R
		Checkpoints_x2 = 1.5 * R + R * np.cos(theta_vec2)
		Checkpoints_y2 = R * np.sin(theta_vec2)
		Checkpoints_x3 = np.linspace(1.5 * R, -1.5*R, n_checkpoints)
		Checkpoints_y3 = R * np.ones(n_checkpoints)
		Checkpoints_x4 = -1.5* R + R * np.cos(theta_vec4)
		Checkpoints_y4 = R * np.sin(theta_vec4)

		xy_lenght =  (6 * R) + (2 * np.pi * R)      # lenght of the xy path
		#amplitude = [1.5 * R]                       # amplitude of the sin wave, i.e. the max height of the path
		amplitude = [1.5 * R , 0.5 * R ]            # amplitude of the sin wave, i.e. the max height of the path
		n_reps = 3                                  # how many complete cycles do we want in a single lap
		n_parts = 4                                 # how many parts do we want the path to be divided into
		offset = 2                                  # offset (z-wise) of the path from the origin
		Checkpoints_x = np.concatenate((Checkpoints_x2[0:n_checkpoints - 1],
				 Checkpoints_x3[0:n_checkpoints - 1],
				 Checkpoints_x4[0:n_checkpoints - 1],
				 Checkpoints_x1[0:n_checkpoints - 1]), axis=0)

		Checkpoints_y = np.concatenate((Checkpoints_y2[0:n_checkpoints - 1],
				 Checkpoints_y3[0:n_checkpoints - 1],
				 Checkpoints_y4[0:n_checkpoints - 1],
				 Checkpoints_y1[0:n_checkpoints - 1]), axis=0)

	if choice == 'savoiardo_long':
		length = 30
		x_shift = 1.0
		y_shift = -0.25
		y_sideways_shift = -2

		R = 0.4 
		theta_init2 = np.pi * -0.5
		theta_end2 = np.pi * 0.5
		theta_vec2 = np.linspace(theta_init2, theta_end2, n_checkpoints)
		theta_init4 = np.pi * 0.5
		theta_end4 = np.pi * 1.5
		theta_vec4 = np.linspace(theta_init4, theta_end4, n_checkpoints)

		Checkpoints_x1 = np.linspace(0, length, n_checkpoints)
		Checkpoints_y1 = np.linspace(0, y_sideways_shift, n_checkpoints) - R

		Checkpoints_x2 = length + R * np.cos(theta_vec2)
		Checkpoints_y2 = R * np.sin(theta_vec2) + y_sideways_shift

		Checkpoints_x3 = np.linspace(length, 0, n_checkpoints)
		Checkpoints_y3 = R * np.ones(n_checkpoints) + np.linspace(y_sideways_shift, 0, n_checkpoints)

		Checkpoints_x4 = 0 + R * np.cos(theta_vec4)
		Checkpoints_y4 = R * np.sin(theta_vec4)



		Checkpoints_x = np.concatenate((Checkpoints_x2[0:n_checkpoints - 1],
				 Checkpoints_x3[2:n_checkpoints - 1],
				 Checkpoints_x4[0:n_checkpoints - 1],
				 Checkpoints_x1[0:n_checkpoints - 1]), axis=0) + x_shift

		Checkpoints_y = np.concatenate((Checkpoints_y2[0:n_checkpoints - 1],
				 Checkpoints_y3[2:n_checkpoints - 1],
				 Checkpoints_y4[0:n_checkpoints - 1],
				 Checkpoints_y1[0:n_checkpoints - 1]), axis=0)+y_shift




	elif choice == 'straight_line_my_house':
		Checkpoints_x = np.linspace(-6, +30, n_checkpoints)
		Checkpoints_y = np.linspace(+0.15, -0.3, n_checkpoints)

	elif choice == 'straight_line_pme':
		Checkpoints_x = np.linspace(-0.1, +30, n_checkpoints)
		Checkpoints_y = np.linspace(+0.0, -0.5, n_checkpoints)

	elif choice == 'straight_line_downstairs':
		Checkpoints_x = np.linspace(-2, +50, n_checkpoints)
		Checkpoints_y = np.linspace(+0.0, -2.5, n_checkpoints)
		
	return Checkpoints_x, Checkpoints_y


# straight path
def straight(xlims, ylims, n_checkpoints, length):
    Checkpoints_x = np.linspace(xlims[0], xlims[1], n_checkpoints)
    Checkpoints_y = np.linspace(ylims[0], ylims[1], n_checkpoints)
    length = length + math.dist([xlims[0], ylims[0]], [xlims[1], ylims[1]])
    return Checkpoints_x, Checkpoints_y, length


# definition of a curve for a path
def curve(centre, R,theta_extremes, n_checkpoints, length):
    theta_init = np.pi * theta_extremes[0]
    theta_end = np.pi * theta_extremes[1]
    theta_vec = np.linspace(theta_init, theta_end, n_checkpoints)
    Checkpoints_x = centre[0] + R * np.cos(theta_vec)
    Checkpoints_y = centre[1] + R * np.sin(theta_vec)
    length = length + abs(theta_end - theta_init) * R

    return Checkpoints_x, Checkpoints_y, length


# This function finds the parameter s of the point on the curve r(s) which provides minimum distance between the path and the vehicle
def find_s_of_closest_point_on_global_path(x_y_state, s_vals_global_path, x_vals_original_path, y_vals_original_path, previous_index, estimated_ds):
	# finds the minimum distance between two consecutive points of the path
	min_ds = np.min(np.diff(s_vals_global_path))
	# find a likely number of idx steps of s_vals_global_path that the vehicle might have travelled
	estimated_index_jumps = math.ceil(estimated_ds / min_ds)
	# define the minimum number of steps that the vehicle could have traveled, if it was moving
	minimum_index_jumps = math.ceil(0.1 / min_ds)

	# in case the vehicle is still, ensure a minimum search space to account for localization error
	if estimated_index_jumps < minimum_index_jumps:
		estimated_index_jumps = minimum_index_jumps

	# define the search space span
	Delta_indexes = estimated_index_jumps * 3

	# takes the previous index (where the vehicle was) and defines a search space around it
	start_i = int(previous_index - Delta_indexes)
	finish_i = int(previous_index + Delta_indexes)

	# check if start_i is negative and finish_i is bigger than the path, in which case the search space is wrapped around
	if start_i < 0:
		s_search_vector = np.concatenate((s_vals_global_path[start_i:], s_vals_global_path[: finish_i]), axis=0)
		x_search_vector = np.concatenate((x_vals_original_path[start_i:], x_vals_original_path[: finish_i]), axis=0)
		y_search_vector = np.concatenate((y_vals_original_path[start_i:], y_vals_original_path[: finish_i]), axis=0)


	elif finish_i > s_vals_global_path.size:
		s_search_vector = np.concatenate((s_vals_global_path[start_i:], s_vals_global_path[: finish_i - s_vals_global_path.size]), axis=0)
		x_search_vector = np.concatenate((x_vals_original_path[start_i:], x_vals_original_path[: finish_i - s_vals_global_path.size]), axis=0)
		y_search_vector = np.concatenate((y_vals_original_path[start_i:], y_vals_original_path[: finish_i - s_vals_global_path.size]), axis=0)

	else:
		s_search_vector = s_vals_global_path[start_i: finish_i]
		x_search_vector = x_vals_original_path[start_i: finish_i]
		y_search_vector = y_vals_original_path[start_i: finish_i]


	# remove the last value to avoid ambiguity since first and last value may be the same
	distances_squared = np.zeros(s_search_vector.size)
	for ii in range(0, s_search_vector.size):
		# evaluate the distance between the vehicle (x_y_state[0:3]) and each point of the search vector (i.e. of the path)
		#distances[ii] = math.dist([x_search_vector[ii], y_search_vector[ii]], x_y_state[0:2])
		distances_squared[ii] = (x_search_vector[ii]-x_y_state[0]) ** 2 + (y_search_vector[ii]-x_y_state[1]) ** 2

	# retrieve the index of the minimum distance element
	local_index = np.argmin(distances_squared)

	# check if the found minimum is on the boarder (indicating that the real minimum is outside of the search vector)
	# this offers some protection against failing the local search but it doesn't fix all of the possible problems
	# for example if path loops back (like a bean shape)
	# then you can still get an error (If you have lane boundary information then you colud put a check on the actual value of the min)
	if local_index == 0 or local_index == s_search_vector.size-1:
		print('search vector was not long enough, doing search on full path')
		distances_squared_2 = np.zeros(s_vals_global_path.size)
		for ii in range(0, s_vals_global_path.size):
			distances_squared_2[ii] = (x_vals_original_path[ii]-x_y_state[0]) ** 2 +  (y_vals_original_path[ii]-x_y_state[1]) ** 2
		index = np.argmin(distances_squared_2)
	else:
		index = np.where(s_vals_global_path == s_search_vector[local_index])
		# extract an int from the "where" operand
		index = index[0]

	s = float(s_vals_global_path[index])
	return s, index



def produce_marker_array_rviz(x, y, rgba, marker_type):
	marker_array = MarkerArray()              # definition of an array of markers
	marker = Marker()                         # single marker within the marker_array

	marker.header.frame_id = "map"            # map frame, used when markers or data should be positioned in relation to a global map.
	marker.header.stamp = rospy.Time.now()    # associate a timestamp to the frame

	# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3 ; Line_strip: 4
	marker.type = marker_type
	marker.id = 0

	# Set the scale of the marker
	marker.scale.x = 0.025
	marker.scale.y = 0.025
	marker.scale.z = 0.025

	# Set the color
	marker.color.r = rgba[0] / 256
	marker.color.g = rgba[1] / 256
	marker.color.b = rgba[2] / 256
	marker.color.a = rgba[3]

	# Set the pose of the marker
	marker.pose.orientation.x = 0.0
	marker.pose.orientation.y = 0.0
	marker.pose.orientation.z = 0.0
	marker.pose.orientation.w = 1.0

	points_list = []
	for i in range(len(x)):
		p = Point()
		p.x = x[i]
		p.y = y[i]
		points_list = points_list + [p]

	marker.points = points_list
	

	# append the created marker to marker_array
	marker_array.markers.append(marker)


	return  marker_array

def produce_marker_rviz(x, y, rgba, marker_type, scale):
	marker = Marker()                         # single marker within the marker_array

	marker.header.frame_id = "map"            # map frame, used when markers or data should be positioned in relation to a global map.
	marker.header.stamp = rospy.Time.now()    # associate a timestamp to the frame

	# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3 ; Line_strip: 4
	marker.type = marker_type
	marker.id = 0

	# Set the scale of the marker
	marker.scale.x = scale
	marker.scale.y = scale
	marker.scale.z = scale

	# Set the color
	marker.color.r = rgba[0] / 256
	marker.color.g = rgba[1] / 256
	marker.color.b = rgba[2] / 256
	marker.color.a = rgba[3]

	# Set the pose of the marker
	marker.pose.orientation.x = 0.0
	marker.pose.orientation.y = 0.0
	marker.pose.orientation.z = 0.0
	marker.pose.orientation.w = 1.0

	# set position
	marker.pose.position.x = x
	marker.pose.position.y = y

	

	return  marker



def evaluate_Fx_2(vx, th):
	#define parameters

	# these are from using the full dataset
	# v_friction = 1.3852415
	# v_friction_static = 0.15652551
	# v_friction_static_tanh_mult = 44.221073
	# v_friction_quad = 0.093703546

	# tau_offset = 0.19454427
	# tau_offset_reverse = 0.26989207
	# tau_steepness = 15.530939
	# tau_steepness_reverse = 10.14599
	# tau_sat_high = 2.3438272
	# tau_sat_high_reverse = 1.4451292

	#these use -1.5 < vx < 1.5 (so low velocity data)
	# v_friction = 0.99375176
	# v_friction_static = 0.36111164
	# v_friction_static_tanh_mult = 39.313255
	# v_friction_quad = 0.2397717

	# tau_offset = 0.20514612
	# tau_offset_reverse = 0.27404037
	# tau_steepness = 13.486378
	# tau_steepness_reverse = 11.863287
	# tau_sat_high = 2.632069
	# tau_sat_high_reverse = 1.4405308


	#fitted from same data as GP for ICRA 2024
	v_friction = 1.0683593
	v_friction_static = 1.1530068
	v_friction_static_tanh_mult = 23.637709
	v_friction_quad = 0.09363517

	tau_offset = 0.16150239
	tau_offset_reverse = 0.16150239
	tau_steepness = 10.7796755
	tau_steepness_reverse = 90
	tau_sat_high = 2.496312
	tau_sat_high_reverse = 5.0




	#friction model
	static_friction = np.tanh(v_friction_static_tanh_mult  * vx) * v_friction_static
	v_contribution = - static_friction - vx * v_friction - np.sign(vx) * vx ** 2 * v_friction_quad 

	#for positive throttle
	th_activation1 = (np.tanh((th - tau_offset) * tau_steepness) + 1) * tau_sat_high
	#for negative throttle
	th_activation2 = (np.tanh((th + tau_offset_reverse) * tau_steepness_reverse)-1) * tau_sat_high_reverse

	throttle_contribution = (th_activation1 + th_activation2) 

	# --------

	Fx = throttle_contribution + v_contribution
	Fx_r = Fx * 0.5
	Fx_f = Fx * 0.5
	return Fx_r + Fx_f


def steer_angle_2_command(steer_angle,car_number):
	# taken on car n. 3! so curves could be different!
	# a = -1.2053807 # the minus sign is correct because if you want to go left you go anti-clockwise so positive steering angle
	# b = 0.38302866
	# c = 0.0 #0.08739186 # this is the offset
	# steer_command = c + 1/a*np.arctan(steer_angle/b)



	if car_number=='1':
		b =  -0.49230292
		c =  -0.01898136
	elif car_number=='2':
		b =  -0.4306743
		c =  -0.0011013203
	elif car_number=='3':
		b =  -0.4677383
		c =  0.013598954

	#steering_angle = self.b * steering_command - self.c
	steer_command = (c + steer_angle)/b

	return steer_command

