#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Float32, Float32MultiArray, Header, Bool
import csv
import datetime
import os
import rospkg
import numpy as np



# Replace 'package_name' with the name of the package you want to locate


class record_platooning_data:
    def __init__(self):

        self.folder_name = '/Data/'
        self.file_name = 'platooning_data_'


        # Initiate this node
        rospy.init_node('platooning_data_recording', anonymous=True)

        # Create new file
        # To later find path to data folder
        self.rospack = rospkg.RosPack()
        date_time = datetime.datetime.now()
        date_time_str = date_time.strftime("%m_%d_%Y_%H_%M_%S")
        file_name = self.rospack.get_path('platooning_utilities') + self.folder_name + self.file_name + date_time_str + '.csv'
        
        file = open(file_name, 'w+') 
        print('saving platooning data to file:  ',file_name)

        # Write header line
        self.writer = csv.writer(file)               
        self.writer.writerow(['elapsed time sensors',
                              'u_des1', 'u_des2', 'u_des3', 'u_des4',
                              'u_com1', 'u_com2', 'u_com3', 'u_com4',
                              'alarm1', 'alarm2', 'alarm3', 'alarm4',
                              'v1', 'v2', 'v3', 'v4',
                              'throttle_1', 'throttle_2', 'throttle_3', 'throttle_4',
                              'acc_imu1', 'acc_imu2', 'acc_imu3', 'acc_imu4',
                              'vrel1', 'vrel2', 'vrel3', 'vrel4',
                              'dist1', 'dist2', 'dist3', 'dist4',
                              'safety_value',
                              'add_ff',
                              'attack',
                              'v_ref'])

        # Initialize variables
        self.u_des1 = 0.0
        self.u_des2 = 0.0
        self.u_des3 = 0.0
        self.u_des4 = 0.0  # Initialize for car 4
        self.u_com1 = 0.0
        self.u_com2 = 0.0
        self.u_com3 = 0.0
        self.u_com4 = 0.0  # Initialize for car 4
        self.v1 = 0.0
        self.v2 = 0.0
        self.v3 = 0.0
        self.v4 = 0.0  # Initialize for car 4
        self.th1 = 0.0
        self.th2 = 0.0
        self.th3 = 0.0
        self.th4 = 0.0  # Initialize for car 4
        self.acc_imu_1 = 0.0
        self.acc_imu_2 = 0.0
        self.acc_imu_3 = 0.0
        self.acc_imu_4 = 0.0  # Initialize for car 4
        self.vrel1 = 0.0
        self.vrel2 = 0.0
        self.vrel3 = 0.0
        self.vrel4 = 0.0  # Initialize for car 4
        self.dist1 = 0.0
        self.dist2 = 0.0
        self.dist3 = 0.0
        self.dist4 = 0.0  # Initialize for car 4
        self.alarm1 = 0.0
        self.alarm2 = 0.0
        self.alarm3 = 0.0
        self.alarm4 = 0.0  # Initialize for car 4
        self.safety_value = 0.0
        self.add_ff_action = False
        self.attack = False
        self.v_ref = 0.0

        # Subscribe to inputs and sensor information topics
        # On-board sensors
        rospy.Subscriber('sensors_and_input_1', Float32MultiArray, self.callback_sensors_and_input_1)
        rospy.Subscriber('sensors_and_input_2', Float32MultiArray, self.callback_sensors_and_input_2)
        rospy.Subscriber('sensors_and_input_3', Float32MultiArray, self.callback_sensors_and_input_3)
        rospy.Subscriber('sensors_and_input_4', Float32MultiArray, self.callback_sensors_and_input_4)  # New subscription for car 4

        # Relative state data
        rospy.Subscriber('relative_state_1', Float32MultiArray, self.callback_rel_state_1)
        rospy.Subscriber('relative_state_2', Float32MultiArray, self.callback_rel_state_2)
        rospy.Subscriber('relative_state_3', Float32MultiArray, self.callback_rel_state_3)
        rospy.Subscriber('relative_state_4', Float32MultiArray, self.callback_rel_state_4)  # New subscription for car 4

        # Safety callback
        rospy.Subscriber('safety_value', Float32, self.callback_safety_val)

        # Control input callback
        rospy.Subscriber('acc_saturated_1', Float32, self.callback_udes1)
        rospy.Subscriber('acc_saturated_2', Float32, self.callback_udes2)
        rospy.Subscriber('acc_saturated_3', Float32, self.callback_udes3)
        rospy.Subscriber('acc_saturated_4', Float32, self.callback_udes4)  # New subscription for car 4

        # Communicated accelerations
        rospy.Subscriber('u_com_1', Float32, self.callback_ucom1)
        rospy.Subscriber('u_com_2', Float32, self.callback_ucom2)
        rospy.Subscriber('u_com_3', Float32, self.callback_ucom3)
        rospy.Subscriber('u_com_4', Float32, self.callback_ucom4)  # New subscription for car 4

        rospy.Subscriber('alarm_1', Float32, self.callback_alarm1)
        rospy.Subscriber('alarm_2', Float32, self.callback_alarm2)
        rospy.Subscriber('alarm_3', Float32, self.callback_alarm3)
        rospy.Subscriber('alarm_4', Float32, self.callback_alarm4)  # New subscription for car 4

        # Add ff callback
        rospy.Subscriber('add_mpc_gamepad', Bool, self.callback_add_mpc_gamepad)
        rospy.Subscriber('attack', Bool, self.callback_attack)

        # V_ref
        rospy.Subscriber('v_ref', Float32, self.callback_v_ref)

        rate = rospy.Rate(10)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():

            current_time = rospy.Time.now()
            elapsed_time = current_time.to_sec() - start_time.to_sec()

            self.writer.writerow([elapsed_time,
                                  self.u_des1, self.u_des2, self.u_des3, self.u_des4,
                                  self.u_com1, self.u_com2, self.u_com3, self.u_com4,
                                  self.alarm1, self.alarm2, self.alarm3, self.alarm4,
                                  self.v1, self.v2, self.v3, self.v4,
                                  self.th1, self.th2, self.th3, self.th4,
                                  self.acc_imu_1, self.acc_imu_2, self.acc_imu_3, self.acc_imu_4,
                                  self.vrel1, self.vrel2, self.vrel3, self.vrel4,
                                  self.dist1, self.dist2, self.dist3, self.dist4,
                                  self.safety_value,
                                  self.add_ff_action,
                                  self.attack,
                                  self.v_ref])
            rate.sleep()

    #state callback functions
    def callback_sensors_and_input_1(self, sensors_and_input_data):
        # sensors_and_input = [elapsed_time, current, voltage, acc_x, acc_y, omega_rad, vel, safety_value,  throttle,  steering]
        array = np.array(sensors_and_input_data.data)
        self.v1 =  array[6]
        self.acc_imu_1 = array[3]
        self.th1 =  array[8]

    def callback_sensors_and_input_2(self, sensors_and_input_data):
        # sensors_and_input = [elapsed_time, current, voltage, acc_x,acc_y, omega_rad, vel]
        array = np.array(sensors_and_input_data.data)
        self.acc_imu_2 = array[3]
        self.v2 =  array[6]
        self.th2 =  array[8]

    def callback_sensors_and_input_3(self, sensors_and_input_data):
        # sensors_and_input = [elapsed_time, current, voltage, acc_x,acc_y, omega_rad, vel]
        array = np.array(sensors_and_input_data.data)
        self.acc_imu_3 = array[3]
        self.v3 =  array[6]
        self.th3 =  array[8]

    # relative state callback functions
    def callback_rel_state_1(self, relsate_msg):
        # sensors_and_input = [vrel,dist]
        self.vrel1 =  np.array(relsate_msg.data)[0]
        self.dist1 =  np.array(relsate_msg.data)[1]

    # relative state callback functions
    def callback_rel_state_2(self, relsate_msg):
        # sensors_and_input = [vrel,dist]
        self.vrel2 =  np.array(relsate_msg.data)[0]
        self.dist2 =  np.array(relsate_msg.data)[1]

    def callback_rel_state_3(self, relsate_msg):
        # sensors_and_input = [vrel,dist]
        self.vrel3 =  np.array(relsate_msg.data)[0]
        self.dist3 =  np.array(relsate_msg.data)[1]

    def callback_safety_val(self, safety_val_msg):
        self.safety_value = safety_val_msg.data

    def callback_udes1(self, msg):
        self.u_des1 = msg.data
    def callback_udes2(self, msg):
        self.u_des2 = msg.data
    def callback_udes3(self, msg):
        self.u_des3 = msg.data

    def callback_ucom1(self, msg):
        self.u_com1 = msg.data
    def callback_ucom2(self, msg):
        self.u_com2 = msg.data
    def callback_ucom3(self, msg):
        self.u_com3 = msg.data

    def callback_alarm1(self, msg):
        self.alarm1 = msg.data
    def callback_alarm2(self, msg):
        self.alarm2 = msg.data
    def callback_alarm3(self, msg):
        self.alarm3 = msg.data


    def callback_add_mpc_gamepad(self, add_ff_action_msg):
        self.add_ff_action = add_ff_action_msg.data

    def callback_attack(self, attack_msg):
        self.attack = attack_msg.data

    def callback_v_ref(self,msg):
        self.v_ref = msg.data


    def callback_sensors_and_input_4(self, sensors_and_input_data):
        # sensors_and_input = [elapsed_time, current, voltage, acc_x, acc_y, omega_rad, vel, safety_value,  throttle,  steering]
        array = np.array(sensors_and_input_data.data)
        self.v4 = array[6]
        self.acc_imu_4 = array[3]
        self.th4 = array[8]

    def callback_rel_state_4(self, relsate_msg):
        # sensors_and_input = [vrel,dist]
        self.vrel4 = np.array(relsate_msg.data)[0]
        self.dist4 = np.array(relsate_msg.data)[1]

    def callback_udes4(self, msg):
        self.u_des4 = msg.data

    def callback_ucom4(self, msg):
        self.u_com4 = msg.data

    def callback_alarm4(self, msg):
        self.alarm4 = msg.data



if __name__ == '__main__':
    try:
        recording = record_platooning_data()

    except rospy.ROSInterruptException:
        print('Something went wrong with setting up topics')