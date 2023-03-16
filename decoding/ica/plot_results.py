import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from utils.analysis import plot
from utils.analysis.tools import chance_level
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

alpha = 0.05



if __name__ == "__main__":
    # list all files in the folder
    files = os.listdir('accuracies')

    # saving accuracies in a dict
    accuracies = {}

    # loop over all files
    for f in files:
        acc = np.load(os.path.join('accuracies', f), allow_pickle=True)
        accuracies[f[:-4]] = acc

    
    # individual plots
    for f, acc in accuracies.items():
        # one fig with the average
        plt = plot.plot_tgm_fig(acc, vmin=20, vmax=80, chance_level=chance_level(588, alpha = alpha, p = 0.5))
        plt.tight_layout()

        out_path = os.path.join('plots', f'tgm_{f[:-4]}.png')
        plt.savefig(out_path)

    # average plot of all with no_ica
    no_ica = np.array([acc for f, acc in accuracies.items() if 'no_ica' in f])
    print(no_ica)
    no_ica = np.mean(no_ica, axis=0)
    plt = plot.plot_tgm_fig(no_ica, vmin=20, vmax=80, chance_level=chance_level(588, alpha = alpha, p = 0.5))
    plt.tight_layout()

    out_path = os.path.join('plots', f'tgm_no_ica.png')
    plt.savefig(out_path)

    # average plot of all with ica
    ica = np.array([acc for f, acc in accuracies.items() if not 'no_ica' in f])
    print(ica)
    ica = np.mean(ica, axis=0)
    plt = plot.plot_tgm_fig(ica, vmin=20, vmax=80, chance_level=chance_level(588, alpha = alpha, p = 0.5))
    plt.tight_layout()

    out_path = os.path.join('plots', f'tgm_ica.png')
    plt.savefig(out_path)

