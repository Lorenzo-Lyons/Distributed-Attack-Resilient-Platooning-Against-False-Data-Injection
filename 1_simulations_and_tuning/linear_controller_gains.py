import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from classes_definintion import platooning_problem_parameters,Vehicle_model, produce_leader_open_loop
from tqdm import tqdm
import tkinter as tk
from tkinter import messagebox

linewidth = 1

# to increase font size for paper figures
font = {'size'   : 32}
matplotlib.rc('font', **font)
params = {'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
matplotlib.rcParams.update(params)
linewidth = 10



dart_parameters = False # True if gain tuning for real robot
select_new_d_h_from_graph = False


# if not clicking on graph selec intervehicle distance d, gains will be set as low as possible in accordance to this distance
if dart_parameters: 
    default_d = 0.5 #[m]
else:
    default_d = 20 #[m]  




# load platooning parameters from class
platooning_problem_parameters_obj = platooning_problem_parameters(dart_parameters)

v_d = platooning_problem_parameters_obj.v_d 
# vehicle maximum acceleration / braking
u_min = platooning_problem_parameters_obj.u_min  
u_max = platooning_problem_parameters_obj.u_max 
v_max = platooning_problem_parameters_obj.v_max 



#- --chosing the gains in the h-d plane ---
# The idea is that we require:
#                              string stability 
#                              critically damped behavior
# Then we add two conditions coming from collision avoidance to determine k,c as a function of h.
# namely: 1- no additional distance travelled during emergency brake, 
#         2-begin braking saturation (even with feedforward action) on the p_rel=d-k/c*v_rel line

from scipy.optimize import root_scalar

# Define your function
def fun(h,d,v_max,u_min,v_d):
    c = v_max/(d-h*v_d)
    k = -u_min/(d-h*v_d)

    val = -((c+h*k)/2-np.sqrt((c+h*k)**2-4*k)/2-k/c )  # minus sign so that the constraint needs to be positive
    return val


def fun_crit_damp(h,d,v_max,u_min,v_d):
    c = v_max/(d-h*v_d)
    k = -u_min/(d-h*v_d)

    val = (c+h*k)**2-4*k
    return val

#define plot ranges
steps = 500
h_min = 0
h_max = 3.5
h_vec = np.linspace(h_min,h_max,steps)

d_crit_damp = np.zeros(steps)
d_stringstab = np.zeros(steps)
for kk in range(steps):
    # Find a zero of the function using the brentq method from SciPy
    h_given = h_vec[kk]
    small_offset = 0.00001
    my_function_crit_damp = lambda x: fun_crit_damp(h_given,x,v_max,u_min,v_d)
    zero_crit_damp = root_scalar(my_function_crit_damp, bracket=[h_given*v_d+small_offset, 1000], method='brentq')
    d_crit_damp[kk] = zero_crit_damp.root

    my_function = lambda x: fun(h_given,x,v_max,u_min,v_d)
    try:
        zero = root_scalar(my_function, bracket=[h_given*v_d+small_offset, zero_crit_damp.root-small_offset], method='brentq')
        d_stringstab[kk] = zero.root
    except:
        pass
    






fig = plt.figure()
fig.subplots_adjust(
top=0.995,
bottom=0.12,
left=0.075,
right=0.995,
hspace=0.2,
wspace=0.2
    )

if select_new_d_h_from_graph:
    plt.title('gain selection figure(click with mouse and close figure)')
else:
    #plt.title('Gain selection figure')
    pass

plt.xlabel('$h$') # ,fontsize=50
plt.ylabel('$d$ [m]') # ,fontsize=50
plt.plot(h_vec,h_vec*v_d,color='slategray',label='$d$ > positive gains constraint',linewidth=linewidth) # ,linewidth=10
plt.plot(h_vec,d_stringstab,color='navy',label='$d$ < string stabilty constraint',linewidth=linewidth) # ,linewidth=10
plt.plot(h_vec,d_crit_damp,color='cornflowerblue',label='$d$ < critically damped constraint',linewidth=linewidth) # ,linewidth=10
plt.fill_between(h_vec, h_vec*v_d, d_stringstab, color='skyblue',
                 alpha=0.25,label='admissible $h-d$ region')
if dart_parameters:
    plt.ylim([0,4])
    plt.xlim([0,1.35])
else:
    plt.ylim([0,90])
    plt.xlim([0,1.5])
plt.legend()



coords = []
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print (f'h = {ix}, d = {iy}')

    global coords
    coords.append((ix, iy))
    plt.plot(ix,iy,'k.')
    plt.draw()

    fig.canvas.mpl_disconnect(cid)

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)







#save the figure
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'gain_tuning.svg'
fig.savefig(image_name, format=image_format, dpi=1200, transparent = True, bbox_inches='tight', pad_inches=0)



if select_new_d_h_from_graph:
    plt.show()
    h=coords[0][0]
    d=coords[0][1]
else:
    d = default_d
    d_idx = (np.abs(d_stringstab - d)).argmin()
    h = h_vec[d_idx]
    #plt.plot(h,d,'k.')


# set k and c according to  the selected h
c = v_max/(d-h*v_d)
k = -u_min/(d-h*v_d)










# --- plotting the d_increment function to have a look --- (this was very useful in the development phase)
# final stopping distance - d (so >0 means collision) when vehicle i (predecessor) brakes at maximum capability and vehicle (i+1)
# starts from the braking saturation line in the v_rel_p_rel plane


from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data
X = np.linspace(0, v_max, 1000) # v_rel
Y = np.linspace(0, v_max, 1000) # v_abs
X, Y = np.meshgrid(X, Y)
Z =(Y/np.abs(u_min)-c*d/(np.abs(u_min)))*X - 0.5/np.abs(u_min)*X**2
#  X*(-c*d/(np.abs(u_min)+u_max)) - 1/(2*np.abs(u_min))*X**2  #
# Find the indices of the maximum value in Z
max_d_increase = np.max(Z)
max_indices = np.argwhere(Z == max_d_increase)
max_v_rel = X[max_indices[0][0],max_indices[0][1]]
max_v_abs = Y[max_indices[0][0],max_indices[0][1]]

#remove parts where the vrel<v constraint is not satisfied
for i in range(1000):
    for t in range(1000):
        if X[i,t] > Y[i,t]:
            Z[i,t] = 0


import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=-np.max(Z), vmax=np.max(Z))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False,norm=norm,zorder=10)



# Plot the maximum point
ax.scatter(max_v_rel, max_v_abs, max_d_increase,s=100, c='k', marker='*',zorder=9)


# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('v_rel')
plt.ylabel('v')
plt.title('Final stopping distance between i and (i+1) vehicles')










# --- check that the system is critically damped --- 
# this should be the case by design but just to be sure we can plot the bode diagram of the transfer function
zero=k/c
delta= np.sqrt((c+k*h)**2-4*k)
poles=(c+k*h)/(2)-delta/2

from scipy import signal
import matplotlib.pyplot as plt

sys = signal.TransferFunction([c, k], [1, c+k*h,k])
w, mag, phase = signal.bode(sys)

plt.figure()
plt.semilogx(w, mag)    # Bode magnitude plot
plt.title(' p_(i+1)/p_i Transfer function : max =' + str(mag.max()))
# plt.figure()
# plt.semilogx(w, phase)  # Bode phase plot




# set maximum u for feedforward action
#u_max_ff = np.abs(u_min)*d/(d-h*v_d) * 0.999


#print out parameters
print('---------------------')
print('Problem parameters:')
print('v_d =',v_d, ' [m/s]')
print('v_max =',v_max, ' [m/s]')
print('u_max =', u_max,' [m/s^2]')
print('u_min =', u_min,' [m/s^2]')
#print('u_max_ff = ',u_max_ff)
print('intervehicle distance lin = ', d)
print(' ')
print('Controller gains:')
print('k =',k)
print('h =',h)
print('c =',c)
print('---------------------')


# save gains
#np.save('saved_linear_controller_gains', np.array([k,c,h,d]))

# show plots
plt.show()



