import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from utils.analysis import plot
from utils.analysis.tools import chance_level
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

alpha = 0.001


# set parameters for all plots
plt.rcParams['font.family'] = 'times new roman'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

def session_plots_individual(accuracies: dict, alpha: float = 0.05, n_trials: int = 588):
    """
    Plots the results of the TGM decoding for each session individually.
    
    Parameters
    ----------
    accuracies: dict
        Dictionary containing the accuracies for each session.
    alpha: float
        Alpha level for the chance level.
    n_trials: int
        Number of trials in the session.
    """
    for f, acc in accuracies.items():
        plt = plot.plot_tgm_fig(acc, vmin=20, vmax=80, chance_level=chance_level(588, alpha = alpha, p = 0.5))
        plt.tight_layout()

        out_path = os.path.join('plots', f'tgm_{f}.png')
        plt.savefig(out_path)
        plt.close()


def main_plot_generator():
    # list all files in the folder
    files = os.listdir('accuracies')

    # saving accuracies in a dict
    accuracies = {}

    # loop over all files
    for f in files:
        acc = np.load(os.path.join('accuracies', f), allow_pickle=True)
        accuracies[f[:-4]] = acc

    # plot all sessions
    session_plots_individual(accuracies)
    
    # Averaged over sessions
    # two subplots one for ica and one for no ica
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # with no_ica
    no_ica = np.array([acc for f, acc in accuracies.items() if 'no_ica' in f])
    no_ica = np.mean(no_ica, axis=0)
    ax[0] = plot.plot_tgm_ax(no_ica, ax[0], vmin=20, vmax=80, chance_level=chance_level(588*11, alpha = alpha, p = 0.5))
    ax[0].set_title('No ICA')

    # with ica
    ica = np.array([acc for f, acc in accuracies.items() if not 'no_ica' in f])
    ica = np.mean(ica, axis=0)
    ax[1] = plot.plot_tgm_ax(ica, ax[1], vmin=20, vmax=80, chance_level=chance_level(588*11, alpha = alpha, p = 0.5))
    ax[1].set_title('ICA')
    
    # add subplot for colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(ax[1].images[0], cax=cax)
    cbar.set_label('Accuracy (%)')

    # making sure to align the plots correctly after adding the colorbar to the other axis
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.axis('off')


    plt.tight_layout()

    out_path = os.path.join('plots', f'average_ica_no_ica.png')
    plt.savefig(out_path)

if __name__ == "__main__":
    main_plot_generator()