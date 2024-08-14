import matplotlib.pyplot as plt
import numpy as np


colours = ['#0063B2FF', '#5DBB63FF']

# set font for all plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.dpi'] = 300

def set_ticks_xy(ax):
    ax.set_xticks(np.arange(0, 251, step=50))
    ax.set_xticklabels([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax.set_yticks(np.arange(0, 251, step=50))
    ax.set_yticklabels([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    
    return ax


def plot_tgm_ax(X, ax, vmin = 30, vmax = 70, chance_level = None, colour_bar = False, title = None, cmap = 'RdBu_r'):
    """
    Plots a time x time generalization matrix on an axis object

    Parameters
    ----------
    X : array, shape (n_time, n_time)
        The generalization matrix to plot.
    ax : matplotlib axis object
        The axis to plot on.

    Returns
    -------
    ax : matplotlib axis object
        The axis with the plot.
    """
    # checks that x has 2 axes with the same length
    if not X.ndim == 2:
        raise ValueError('X must be 2D')
    
    if not X.shape[0] == X.shape[1]:
        raise ValueError('X must be square')

    im = ax.imshow(X*100, vmin = vmin, vmax = vmax, origin = 'lower', cmap = cmap)
    
    if chance_level is not None:
        ax.contour(X*100, levels=[chance_level*100], colors='gray', alpha = 0.5, linewidths=0.5)

    ax = set_ticks_xy(ax)
    
    if colour_bar:
        cb = plt.colorbar(im, ax = ax, location = 'top', shrink = 0.5)
        cb.set_label(label = 'Accuracy (%)')

    if title:
        ax.set_title(title)
    
    return ax



def plot_tgm_fig(X, vmin = 30, vmax = 70, savepath = None, chance_level = None, title = None, cbar_loc = "top", linewitdh = 0.5, cmap = 'RdBu_r'):
    if not X.shape == (250, 250):
        raise ValueError('X must be 250, 250')

    fig, ax = plt.subplots(1, 1, figsize = (7, 7), dpi = 400)

    im = ax.imshow(X*100, vmin = vmin, vmax = vmax, origin = 'lower', cmap = cmap)
    if chance_level is not None:
        plt.contour(X*100, levels=[chance_level*100], colors='k', alpha = 0.5, linewidths=linewitdh)


    ax = set_ticks_xy(ax)

    cb = plt.colorbar(im, ax = ax, location = cbar_loc, shrink = 0.5)
    cb.set_label(label = 'Accuracy (%)')
    
    if cbar_loc == "right":
        # turn the accuracy label
        cb.ax.yaxis.label.set_rotation(270)
        #add padding
        cb.ax.yaxis.labelpad = 20
        
    if title:
        ax.set_title(title)

    ax.set_xlabel('Test time (s)')
    ax.set_ylabel('Train time (s)')
    
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
    
    return plt