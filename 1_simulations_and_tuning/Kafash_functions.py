# Create a unit circle and transform it to an ellipse defined by Y_proj_inv
import numpy as np
from matplotlib.patches import Ellipse
def plot_ellipse(P, m, ax, **kwargs):
    """
    Plot the ellipse defined by x^T P x = m
    """
    cov = np.linalg.inv(P)  # Convert from P to covariance shape
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Ellipse orientation from largest eigenvector
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # Scale axes by sqrt(m * eigenvalue)
    width, height = 2 * np.sqrt(m * vals)

    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                       **kwargs)
    ax.add_patch(ellipse)

    ax.set_aspect('equal')
    ax.grid(True)


def plot_trajectories_timeseries(axs,time,x_history,ellipse_val,scale,add_label,attack_mitigation):
    colors = ['cadetblue', 'coral', 'teal','plum']
    colors_atck_mit_ON = ['dodgerblue', 'orangered', 'forestgreen', 'purple']

    if add_label:
        label_d1 = 'd1'
        label_d2 = 'd2'
        label_v0 = 'v0'
        label_v1 = 'v1'
        label_v2 = 'v2'
        label_ellipse = 'Ellipse Value'
        label_max_val = 'max_val'
        if attack_mitigation: # add atck_mit_ON to the label
            label_d1 = 'd1 attack-mitigation ON'
            label_d2 = 'd2 attack-mitigation ON'
            label_v0 = 'v0 attack-mitigation ON'
            label_v1 = 'v1 attack-mitigation ON'
            label_v2 = 'v2 attack-mitigation ON'
            label_ellipse = 'Ellipse Value attack-mitigation ON'
            colors = colors_atck_mit_ON
    else:
        label_d1 = ''
        label_d2 = ''
        label_v0 = ''
        label_v1 = ''
        label_v2 = ''
        label_ellipse = ''
        label_max_val = ''


    # First subplot: d1, d2
    axs[0].plot(time, x_history[:, 0], label=label_d1,color=colors[1])
    axs[0].plot(time, x_history[:, 1], label=label_d2,color=colors[2])
    axs[0].set_ylabel('Distance States')
    axs[0].set_title('Distance State Evolution')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: v0, v1, v2
    axs[1].plot(time, x_history[:, 2], label=label_v0,color=colors[0])
    axs[1].plot(time, x_history[:, 3], label=label_v1,color=colors[1])
    axs[1].plot(time, x_history[:, 4], label=label_v2,color=colors[2])
    axs[1].set_ylabel('Velocity States')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_title('Velocity State Evolution')
    axs[1].legend()
    axs[1].grid(True)

    # plot ellipse val
    axs[2].plot(time, scale * np.ones(len(time)), label=label_max_val,color='gray',linestyle='--')
    axs[2].plot(time, ellipse_val, label=label_ellipse,color=colors[3])
    axs[2].set_ylabel('Ellipse Value')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_title('Ellipse Value = x^T P x, max value should be sqrt(m)')
    axs[2].legend()

import cvxpy as cp
def reachable_set_given_bounds(a,F,G,R):
    n = F.shape[1]  # Number of states

    # Define variable
    P = cp.Variable((n, n), symmetric=True)

    # Build F(P) as a CVXPY expression
    Q_expr = cp.bmat([
        [a * P - F.T @ P @ F,     -F.T @ P @ G],
        [-G.T @ P @ F, (1 - a) * R - G.T @ P @ G]
    ])
    # np.transpose(
    # Constraints
    constraints = [
        P-0.0001 * np.eye(P.shape[0]) >> 0,    # P is positive definite
        Q_expr >> 0      # Q is positive semidefinite
    ]

    # Objective
    objective = cp.Minimize(-cp.log_det(P))

    # Problem definition and solving
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False) 
    except:
        print('usind SCS solver instead of CLARABEL')
        prob.solve(solver=cp.SCS, verbose=False)
    print('')
    print('solved problem 8')
    print("Problem status:", prob.status)
    print('objective value:', prob.value)

    return P.value, prob


def evaluate_new_bounds_with_constraints(a,F,G,R,c1,c2,b1,b2):
    n = F.shape[1]  # Number of states
    m = G.shape[1]  # Number of inputs

    # Optimization variables
    Y = cp.Variable((n, n), symmetric=True)
    R_hat_diag = cp.Variable(m)  # diagonal entries
    R_hat = cp.diag(R_hat_diag)

    # using Laura's solution
    #R_hat = np.diag([35.3950, 19.5614, 35.3950]) # this is the matrix that we will use to define the ellips



    # Define constraints
    constraints = [
        R_hat >> R,
        Y - 0.001 * np.eye(n)>> 0,
        cp.quad_form(c1, Y) <= (b1 ** 2) / m,
        cp.quad_form(c2, Y) <= (b2 ** 2) / m,
        cp.bmat([
            [a * Y,             np.zeros((n, m)), Y @ F.T],
            [np.zeros((m, n)), (1 - a) * R_hat,   G.T],
            [F @ Y,             G,                Y]
        ]) >> 0
    ]

    # Solve the SDP
    prob = cp.Problem(cp.Minimize(cp.trace(R_hat)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL) # , max_iters=1000
    except:
        print('usind SCS solver instead of CLARABEL')
        prob.solve(solver=cp.SCS, verbose=False)
    print('')
    print('solving problem 13 - new bounds with constraints')
    print("Problem status:", prob.status)
    print('objective value:', prob.value)
    # evaluate eigenvalues of Y
    eigvals = np.linalg.eigvals(Y.value)
    # print with 2 decimal points
    print("Eigenvalues of Y: [", ", ".join(f"{val:.3f}" for val in eigvals), "]")

    
    return Y.value, R_hat.value




