import numpy as np


class platooning_problem_parameters():
    def __init__(self,dart):
        if dart:
            # select parameters for real robots
            self.v_d = 1 # desired velocity in m/s

            # vehicle maximum acceleration / braking
            self.u_min = -1.0 # m/s^2  # the minimum requirement is 0.6 and a lot of trucks are higher than than so ok
            self.u_max = +1.0 # m/s^2
            self.v_max = 1.4 #m/s  # so this is quite realistic for heavy trucks on a motorway

        else:
            # select parameters
            self.v_d = 90 # desired velocity in Km/h As a refrence most countries are 90km, only italy is 100, and some are 80.

            # vehicle maximum acceleration / braking
            self.u_min = -0.8 # [g] m/s^2  # the minimum requirement is 0.6 and a lot of trucks are higher than than so ok
            self.u_max = +0.5 # [g] m/s^2
            self.v_max = 100 #Km/h  # so this is quite realistic for heavy trucks on a motorway

            # convert everything to standard units [m], [m/s], [m/s^2]
            self.v_d = self.v_d * 1000 / 3600
            self.u_min = self.u_min * 9.81
            self.u_max = self.u_max * 9.81
            self.v_max = self.v_max * 1000 / 3600





class Vehicle_model():
    def __init__(self,vehicle_number,x0,v0,leader,controller_parameters,vehicle_parameters):

        #set initial state
        self.x = x0
        self.v = v0

        #set vehicle number
        self.vehicle_number = vehicle_number
        #set leader
        self.leader = leader

        #set controller parameters
        self.v_d = controller_parameters[0]
        self.d = controller_parameters[1]
        self.k = controller_parameters[2]
        self.h = controller_parameters[3]
        self.c = controller_parameters[4]

        #set vehicle parameters
        self.v_max = vehicle_parameters[0]
        self.v_min = 0.0
        self.u_max = vehicle_parameters[1]
        self.u_min = vehicle_parameters[2]

        #prepare to record mpc solution
        self.v_vec_mpc = 0
        self.x_vec_mpc = 0
        self.u_mpc = 0
        self.u_lin = 0
        self.u = 0
    
    def compute_control_action(self,x_leader,v_leader):
        Dp = self.x - x_leader 
        Dv = self.v - v_leader 

        #linear controller action
        u_lin = -self.k*(Dp + self.d) - self.k*self.h*(self.v-self.v_d) - self.c*(Dv)

        return u_lin
    
    def integrate_state(self,u,dt_int):
        candidate_new_v = self.v + u * dt_int

        if candidate_new_v >= self.v_max:
            # do not increase the speed any further
            self.v = self.v_max
            self.x = self.x + self.v * dt_int

        elif candidate_new_v <= self.v_min:
            # do not decrease the speed any further
            self.v = self.v_min
            self.x = self.x + self.v * dt_int

        else:
            self.x = self.x + self.v * dt_int
            self.v = candidate_new_v



def set_scenario_parameters(scenario,d,v_d,c,k,h,v_max,u_min,u_max):

    if scenario==1:
        #follower initial position (v=v_max)
        v_rel_follower_1 = -5
        p_rel_1 = -d/2  
        #because otherwise all followers will have smaller velocity with respect to the leader
        v_rel_follower_others = 0
        p_rel_others = 0

        # Leader
        x0_leader = 0
        v0_leader = v_d

        #use u_ff?
        use_ff = False

        # leader acceleration function
        leader_acc_fun = lambda t: 0

    elif scenario==2:
        #follower initial position (v=v_max)
        v_rel_follower_1 = 0
        p_rel_1 = 0

        #because otherwise all followers will have smaller velocity with respect to the leader
        v_rel_follower_others = 0
        p_rel_others = 0

        # Leader
        x0_leader = 0
        v0_leader = v_d

        #use u_ff?    
        use_ff = False

        # leader acceleration function
        period = 10 # [s]
        amplitude = 4 #[m/s^2]
        leader_acc_fun = lambda t: np.sin(t/period*2*np.pi) * amplitude



    elif scenario==3:
            #follower initial position (v=v_max)
        v_rel_follower_1 = 0
        p_rel_1 = 0

        #because otherwise all followers will have smaller velocity with respect to the leader
        v_rel_follower_others = 0
        p_rel_others = 0

        # Leader
        x0_leader = 0
        v0_leader = v_d

        #use u_ff?    
        use_ff = True

        # leader acceleration function
        period = 10 # [s]
        amplitude = 4 #[m/s^2]
        leader_acc_fun = lambda t: np.sin(t/period*2*np.pi) * amplitude

    elif scenario==4:
            #follower initial position (v=v_max)
        v_rel_follower_1 = 0
        p_rel_1 = 0

        #because otherwise all followers will have smaller velocity with respect to the leader
        v_rel_follower_others = 0
        p_rel_others = 0

        # Leader
        x0_leader = 0
        v0_leader = v_d

        #use u_ff?    
        use_ff = True

        # leader acceleration function
        period = 10 # [s]
        amplitude = 4 #[m/s^2]
        leader_acc_fun = lambda t: np.sin(t/period*2*np.pi) * amplitude


    elif scenario==5:

        #follower initial position (v=v_max)
        v_rel_follower_1 = 0
        p_rel_1 = 0

        #because otherwise all followers will have smaller velocity with respect to the leader
        v_rel_follower_others = 0
        p_rel_others = 0

        # Leader
        x0_leader = 0
        v0_leader = v_d

        #use u_ff?    
        use_ff = True

        # leader acceleration function
        leader_acc_fun = lambda t: 0

        # attack function
        period = 10 # [s]
        amplitude = 4 #[m/s^2]
        attack_function = lambda t,u_i: np.sin(t/period*2*np.pi) * amplitude


    elif scenario==6:

        #follower initial position (v=v_max)
        v_rel_follower_1 = 0
        p_rel_1 = 0

        #because otherwise all followers will have smaller velocity with respect to the leader
        v_rel_follower_others = 0
        p_rel_others = 0

        # Leader
        x0_leader = 0
        v0_leader = v_d

        #use u_ff?    
        use_ff = True

        # leader acceleration function
        leader_acc_fun = lambda t: 0

        # attack function
        attack_function = lambda t,u_i: u_max * 1000 # extremely high value

    elif scenario==7:

        #follower initial position (v=v_max)
        v_rel_follower_1 = 0
        p_rel_1 = 0

        #because otherwise all followers will have smaller velocity with respect to the leader
        v_rel_follower_others = 0
        p_rel_others = 0

        # Leader
        x0_leader = 0
        v0_leader = v_d

        #use u_ff?    
        use_ff = True

        # leader acceleration function
        time_to_brake = 100 #simulation should be 200 s long
        leader_acc_fun = lambda t: 0 if t < time_to_brake  else u_min

        # attack function
        attack_function = lambda t,u_i: u_max * 1000 # extremely high value


    if 'attack_function' not in locals():  
        attack_function = []


    return v_rel_follower_1,p_rel_1,v_rel_follower_others,p_rel_others,\
            x0_leader,v0_leader,\
            leader_acc_fun,use_ff,attack_function










def produce_leader_open_loop(leader_model,u_leader_vec,dt):
    N = len(u_leader_vec)
    #forwards integrate leader state to provide as the mpc output
    v_vec_mpc = np.zeros(N)
    x_vec_mpc = np.zeros(N)

    # set current state as first state in the open loop prediction
    v_vec_mpc[0]=leader_model.v
    x_vec_mpc[0]=leader_model.x

    # compute open loop predction of the leader
    for jj in range(1,N):
        x_vec_mpc[jj]=x_vec_mpc[jj-1] + v_vec_mpc[jj-1] * dt
        v_vec_mpc[jj]=v_vec_mpc[jj-1] + u_leader_vec[jj-1] * dt

    # store results
    leader_model.v_vec_mpc = v_vec_mpc
    leader_model.x_vec_mpc = x_vec_mpc
    leader_model.u_vec_long = u_leader_vec
    #leader_model.u_mpc = u_mpc_leader
    leader_model.adjustment_vector = np.zeros(N)
    #u = u_mpc_leader # set overall u
    return leader_model














