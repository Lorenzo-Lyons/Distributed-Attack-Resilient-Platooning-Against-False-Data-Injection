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