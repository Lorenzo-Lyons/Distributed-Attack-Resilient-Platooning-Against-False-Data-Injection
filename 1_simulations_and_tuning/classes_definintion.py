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


class DMPC():
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    

    def __init__(self):
        pass

    
    def export_double_integrator_model(self):
        from casadi import SX, vertcat
        from acados_template import AcadosModel
        model_name = "double_integrator"

        # States & controls
        vel = SX.sym('v')  # velocity
        pos = SX.sym('x')  # position

        #x = vertcat(vel,pos)  # state (position)
        u = SX.sym('u')  # control input (velocity)

        p = vertcat(SX.sym('x_ref'), SX.sym('x_assumed_self'))  # reference for the state (stage-wise)

        # Dynamics (continuous-time)
        f_expl = vertcat(u, vel)

        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = vertcat(vel,pos)
        model.u = u
        model.name = model_name

        # Set parameters
        model.p = p

        return model

    # 2. Define the MPC solver
    def setup_mpc_solver(self,Tf, N, u_max,u_min,v_max,v_min):
        from acados_template import AcadosOcp, AcadosOcpSolver
        import os

        solver_file = "acados_ocp_platooning.json"

        # Create ocp object
        ocp = AcadosOcp()

        # Export model
        model = self.export_double_integrator_model()

        # Set model
        ocp.model = model

        # Set dimensions
        nx = 2  # state dimension
        nu = 1  # input dimension
        npar = 2  # number of parameters
        #ny = nx + nu  # number of outputs in cost function

        # Set prediction horizon
        ocp.dims.N = N

        # 1. Set cost function
        qu = 1  # Control weight
        qx = 1 #1 #1.6 # 1.6 for immediate crash in FDI  10 for crash during emergency brake

        qx_assumed = 20
        qx_final = 10000 # this needs to be very high because it should really be a hard constraint in theory

        # The 'EXTERNAL' cost type can be used to define general cost terms
        # We'll modify the cost to account for stage-wise references
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # External cost expressions
        x_ref = model.p[0]   # reference for the state (zero velocity)
        x_assumed_self = model.p[1] # position comunicated to neighbours on the previous iteration

        ocp.model.cost_expr_ext_cost =  qu * model.u**2 + qx * (model.x[1]-x_ref)**2 + qx_assumed * (model.x[1]-x_assumed_self)**2 # + qx * (model.x[1]-x_ref)**2 #
        ocp.model.cost_expr_ext_cost_e =  qx_final * (model.x[1] - x_ref)**2 


        # 2. Set constraints

        # Set the state constraints bounds
        ocp.constraints.lbx = np.array([v_min])  # Lower bound on velocity
        ocp.constraints.ubx = np.array([v_max])  # Upper bound on velocity
        ocp.constraints.idxbx = np.array([0])    # Specify which state is constrained (0 for velocity)


        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.lbu = np.array([u_min])
        ocp.constraints.ubu = np.array([-u_min])  # this should really be u_max, but to have the same attack as in other cases we set it to -u_min
        ocp.constraints.idxbu = np.array([0])

        # # Enforce terminal control action to be zero at the last stage
        ocp.constraints.lbu_e = np.array([0.0])  # Lower bound on terminal input
        ocp.constraints.ubu_e = np.array([0.0])  # Upper bound on terminal input

        # Initial state constraint
        ocp.constraints.x0 = np.array([0.0, 0.0])  # Starting from position 0

        # 3. Set solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.hessian_approx = 'EXACT' # EXACT   may fix the warning
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.tf = Tf

        # Initialize parameters with default values (this step is important to avoid dimension mismatch)
        ocp.parameter_values = np.zeros(npar)

        # Create solver
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_file)

        return acados_ocp_solver
    
    def set_up_sovler_iteration(self, solver,N,x_ref,x_open_loop_prev,v_open_loop_prev_N,dt,u_min,u_max,x_current):
        # store reference trajectory
        x_assumed_open_loop = np.zeros(N+1)
    
        # Set stage-wise references and constraints
        for k in range(N+1):
            x_ref_k = x_ref[k]  # Stage-wise references
            
            #if i > 0:
            if k == N:
                x_assumed_self = x_open_loop_prev[k] + dt * v_open_loop_prev_N  # this can be done since the acceleration is constrained to 0 at last stage
            else:
                x_assumed_self = x_open_loop_prev[k+1] # position comunicated to neighbours on the previous iteration (time shifted by 1)
            # else:
            #     x_assumed_self = x_ref[i+k] # in the first stage use the reference as the past communicated position

            p_array = np.array([x_ref_k, x_assumed_self]) # each stage knows what the final reference is
            solver.set(k, 'p', p_array)  # Set stage-wise references

            # store reference trajectory
            x_assumed_open_loop[k] = x_assumed_self
        



        # set initial condition
        solver.set(0, "lbx", x_current)
        solver.set(0, "ubx", x_current)

        return solver,x_assumed_open_loop

    def solve_mpc(self,solver,N):
        # Solve MPC problem
        status = solver.solve()

        # Get optimal outputs
        u_open_loop = np.zeros(N)
        v_open_loop = np.zeros((N+1))
        x_open_loop = np.zeros((N+1))

        # collect open loop trajectories
        for k in range(N+1):
            # Get predicted states
            state = solver.get(k, "x")
            v_open_loop[k] = state[0]
            x_open_loop[k] = state[1]

        for k in range(N):
            u_open_loop[k] = solver.get(k, "u")  

        return u_open_loop,v_open_loop,x_open_loop
    

    def set_initial_guess(self,solver, v_guess , x_guess, u_guess):
        N = solver.acados_ocp.dims.N  # Length of the horizon

        # Set initial guess for the entire horizon
        for i in range(N):
            solver.set(i, "x", np.array([v_guess[i],x_guess[i]]))
            solver.set(i, "u", u_guess[i])

        # Set the terminal state guess (stage N)
        solver.set(N, "x", np.array([v_guess[N],x_guess[N]]))
        return solver




class Vehicle_model():
    def __init__(self,vehicle_number,x0,v0,leader,controller_parameters,vehicle_parameters,use_MPC,dt_int):

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

        # #set up MPC solver if needed
        # if use_MPC:
        #     # default MPC parameters
        #     self.dt_int = dt_int # Time step [s]
        #     self.N = 20  # Number of steps in the horizon
        #     Tf = dt_int * self.N  # Time horizon

        #     self.DMPC_obj = DMPC()
        #     self.solver = self.DMPC_obj.setup_mpc_solver(Tf, self.N,self.u_max,self.u_min,self.v_max,self.v_min)
            
    
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
    

    period = 10 # [s]
    amplitude = 1.5
    initial_phase = -0.5*np.pi

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

        #use MPC?
        use_MPC = False
        use_baseline_linear = False

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
        leader_acc_fun = lambda t0,stage:  np.sin(t0/period*2*np.pi+initial_phase) * amplitude

        #use MPC?
        use_MPC = False
        use_baseline_linear = False

        # dummy values
        time_to_brake = 0
        time_to_attack = 0

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
        leader_acc_fun = lambda t:  np.sin(t/period*2*np.pi+initial_phase) * amplitude

        #use MPC?
        use_MPC = False
        use_baseline_linear = False

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
        leader_acc_fun = lambda t:  np.sin(t/period*2*np.pi+initial_phase) * amplitude

        #use MPC?
        use_MPC = False
        use_baseline_linear = False

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
        attack_function = lambda t,u_i: np.sin(t/period*2*np.pi+initial_phase) * amplitude

        #use MPC?
        use_MPC = False
        use_baseline_linear = False

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

        #use MPC?
        use_MPC = False
        use_baseline_linear = False

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
        time_to_brake = 11
        leader_acc_fun = lambda t0, t_stage: 0 if t0 < time_to_brake  else u_min 

        # attack function
        time_to_attack = 1
        attack_function = lambda t0,u_i: u_max if t0 > time_to_attack  else leader_acc_fun(t0,0) # extremely high value for attack

        #use MPC?
        use_MPC = False
        use_baseline_linear = False


    elif scenario==8:
        #follower initial position (v=v_max)
        v_rel_follower_1 = 0
        p_rel_1 = +d/2  
        #because otherwise all followers will have smaller velocity with respect to the leader
        v_rel_follower_others = 0
        p_rel_others = 0

        # Leader
        x0_leader = 0
        v0_leader = v_d

        #use u_ff?
        use_ff = False

        # leader acceleration function
        leader_acc_fun = lambda t0, t_stage: 0

        #use MPC?
        use_MPC = True
        use_baseline_linear = False

        # dummy values
        time_to_brake = 0
        time_to_attack = 0

    elif scenario==9:
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
       
        leader_acc_fun = lambda t0, t_stage:  np.sin(t_stage/period*2*np.pi+initial_phase) * amplitude

        #use MPC?
        use_MPC = True
        use_baseline_linear = False

        time_to_brake = 0 # dummy value
        time_to_attack = 0 # dummy value


    elif scenario==10:

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
        time_to_brake = 11
        leader_acc_fun = lambda t0, t_stage: 0 if t0 < time_to_brake  else u_min

        # attack function
        time_to_attack = 1
        attack_function = lambda t: u_max
        #attack_function = lambda t: u_max*0.1 

        #use MPC?
        use_MPC = True
        use_baseline_linear = False

    elif scenario==11:

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
        time_to_brake = 2
        leader_acc_fun = lambda t: 0 if t < time_to_brake  else u_min

        # attack function
        #attack_function = lambda t: u_max  # extremely high value

        #use MPC?
        use_MPC = False
        use_baseline_linear = False
        # dummy vaslues
        time_to_attack = 0

    elif scenario==12:

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
        time_to_brake = 25
        leader_acc_fun = lambda t0, t_stage: 0 if t0 < time_to_brake  else u_min

        # attack function
        #attack_function = lambda t: u_max  # extremely high value

        #use MPC?
        use_MPC = True
        use_baseline_linear = False

    elif scenario==13:

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
        
        leader_acc_fun = lambda t0, t_stage:  np.sin(t0/period*2*np.pi+initial_phase) * amplitude

        # attack function
        #attack_function = lambda t: u_max  # extremely high value

        #use MPC?
        use_MPC = False
        use_baseline_linear = True


    elif scenario==14:

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
        time_to_brake = 2
        leader_acc_fun = lambda t0, t_stage: 0 if t0 < time_to_brake  else u_min

        # attack function
        #attack_function = lambda t: u_max  # extremely high value

        #use MPC?
        use_MPC = False
        use_baseline_linear = True

        #dummy values
        time_to_attack = 0



    if 'attack_function' not in locals():  
        attack_function = []


    return v_rel_follower_1,p_rel_1,v_rel_follower_others,p_rel_others,\
            x0_leader,v0_leader,\
            leader_acc_fun,use_ff,attack_function,\
            use_MPC,use_baseline_linear,time_to_brake,time_to_attack










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

# support function to generate color gradients for plots
import matplotlib.colors as mcolors
import numpy as np

def hex_to_rgb(hex_color):
    # Convert a hex color to an RGB tuple
    return mcolors.hex2color(hex_color)

def rgb_to_hex(rgb_color):
    # Convert an RGB tuple to a hex color
    return mcolors.to_hex(rgb_color)

def generate_color_gradient(n, start_color, end_color):
    # Convert the start and end hex colors to RGB tuples
    start_rgb = np.array(hex_to_rgb(start_color))
    end_rgb = np.array(hex_to_rgb(end_color))
    
    # Prepare an empty list to store the gradient colors
    gradient = []
    
    # Generate 'n' colors by linear interpolation between start and end colors
    for t in np.linspace(0, 1, n):
        interpolated_color = (1 - t) * start_rgb + t * end_rgb
        gradient.append(rgb_to_hex(interpolated_color))
    
    return gradient


# Helper functions for hex and RGB conversion
def hex_to_rgb2(hex_color):
    """Convert hex color (#RRGGBB) to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex2(rgb_color):
    """Convert RGB tuple to hex color (#RRGGBB)."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

def generate_color_1st_last_gray(n, start_color, end_color):
    # Convert the start and end hex colors to RGB tuples
    start_rgb = np.array(hex_to_rgb2(start_color))
    end_rgb = np.array(hex_to_rgb2(end_color))
    
    # Define the gray color (RGB value)
    gray_rgb = np.array([128, 128, 128])  # A mid-gray

    # Prepare an empty list to store the gradient colors
    gradient = []

    # First color is the start color
    gradient.append(rgb_to_hex2(start_rgb))

    # Middle colors are gray
    for i in range(n - 2):  # n - 2 middle colors (because 1 start and 1 end color)
        gradient.append(rgb_to_hex2(gray_rgb))
    
    # Last color is the end color
    gradient.append(rgb_to_hex2(end_rgb))

    return gradient











