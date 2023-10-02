import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from utils.analysis import plot
from utils.analysis.tools import chance_level

import matplotlib.pyplot as plt
import numpy as np
import os

# set parameters for all plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    acc = np.load(os.path.join('accuracies', 'accuracies_memory_or_visual.npy'), allow_pickle=True)

    # one fig with the average
    plt = plot.plot_tgm_fig(acc, vmin=25, vmax=75, chance_level=chance_level(6146, alpha = 0.001, p = 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'tgm_decoding_condition.png'))
    
    