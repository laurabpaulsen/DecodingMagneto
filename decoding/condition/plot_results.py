import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from utils.analysis import plot
from utils.analysis.tools import chance_level

import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    acc = np.load(os.path.join('accuracies', 'accuracies_memory_or_visual.npy'), allow_pickle=True)

    # one fig with the average
    plt = plot.plot_tgm_fig(acc, vmin=20, vmax=80, chance_level=chance_level(13190, alpha = 0.001, p = 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'tgm_decoding_condition.png'))
    
    