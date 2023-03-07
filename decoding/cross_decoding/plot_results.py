"""
dev notes:
- [ ] Get correct number of trials for vis and mem
"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from utils.analysis import plot
from utils.analysis.tools import chance_level
import matplotlib.pyplot as plt
import numpy as np
import os


for parc in ['aparc', 'aparc.a2009s','aparc.DKTatlas', 'sens', 'HCPMMP1']:
    # read in results
    acc = np.load(os.path.join('accuracies', f'cross_decoding_10_LDA_{parc}.npy'), allow_pickle=True)

    fig, axs = plt.subplots(acc.shape[0], acc.shape[1], figsize = (30, 30))
    for idx, i in enumerate(range(acc.shape[0])):
        for jdx, j in enumerate(range(acc.shape[1])):
            axs[i, j] = plot.plot_tgm_ax(acc[i, j], ax=axs[i, j], vmin=35, vmax=65)
            axs[i, j].set_title(f'train:{i+1}, test:{j+1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'cross_decoding_{parc}.png'))
    plt.close()

    # one fig with the average
    plt = plot.plot_tgm_fig(acc.mean(axis=(0, 1)), vmin=45, vmax=55, chance_level=chance_level(8039, alpha = 0.001, p = 0.5))
    plt.savefig(os.path.join('plots', f'cross_decoding_{parc}_average.png'))
    
    # a plot with the average of 4 different things (test mem + train mem, test vis + train vis, test mem + train vis, test vis + train mem)
    fig, axs = plt.subplots(2, 3, figsize = (10, 10))
    vis = np.array([0, 1, 2, 3, 5, 6])
    mem = np.array([7, 8, 9, 10])
    
    n_trials_vis = 5184 # GET THE CORRECT NUMBER
    n_trials_mem = 3186# GET THE CORRECT NUMBER
    
    
    axs[0, 0] = plot.plot_tgm_ax(acc[vis, vis, :, :].mean(axis = 0), ax=axs[0, 0], vmin=40, vmax=60, chance_level=chance_level(n_trials_vis, alpha = 0.001, p = 0.5))
    axs[0, 0].set_title('train: vis,  test:vis')

    axs[0, 1] = plot.plot_tgm_ax(acc[mem, mem, :, :].mean(axis = 0), ax=axs[0, 1], vmin=40, vmax=60, chance_level=chance_level(n_trials_mem, alpha = 0.001, p = 0.5))
    axs[0, 1].set_title('train:mem, test:mem')

    axs[1, 0] = plot.plot_tgm_ax(acc[vis,:, :, :][:, mem, :, :].mean(axis = (0, 1)), ax=axs[1, 0], vmin=45, vmax=55, chance_level=chance_level(n_trials_mem, alpha = 0.001, p = 0.5))
    axs[1, 0].set_title('train:vis, test:mem')

    axs[1, 1] = plot.plot_tgm_ax(acc[mem, :, :, :][:, vis, :, :].mean(axis = (0, 1)), ax=axs[1, 1], vmin=45, vmax=55, chance_level=chance_level(n_trials_vis, alpha = 0.001, p = 0.5))
    axs[1, 1].set_title('train:mem, test:vis')

    # plot colourbar
    axs[0, 2].axis('off')
    axs[1, 2].axis('off')
    fig.colorbar(axs[0, 0].images[0], ax=axs[0, 2], fraction=0.046, pad=0.04)
    fig.colorbar(axs[1, 0].images[0], ax=axs[1, 2], fraction=0.046, pad=0.04)

    plt.savefig(os.path.join('plots', f'cross_decoding_{parc}_average_vis_mem.png'))




