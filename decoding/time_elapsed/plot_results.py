import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported



colours = ['#0063B2FF', '#5DBB63FF']

# set font for all plots
plt.rcParams['font.family'] = 'sans-serif'
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


def plot_squared_error_tgm(tgm, savepath:Path = None):

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # plot the results
    im = ax.imshow(tgm, origin='lower', cmap="RdBu")

    # add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    # add title to colorbar
    cbar.set_label('Mean Squared Error')

    ax.set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])


    if savepath:
        plt.savefig(savepath)

    


if __name__ == "__main__":
    path = Path(__file__)


    ### PLOTS OF PREDICTING THE SESSION NUMBER ###
    tgm_files = ['animate_combined_predict_session_number.npy', 'inanimate_combined_predict_session_number.npy', 'animate_memory_predict_session_number.npy', 'inanimate_memory_predict_session_number.npy', 'animate_visual_predict_session_number.npy', 'inanimate_visual_predict_session_number.npy']

    for file in tgm_files:
        try:
            tgm = np.load(path.parent / 'results' / file, allow_pickle=True)
            
            # calculate the mean squared error across the 2 axis (contains the error)
            tgm = np.mean(np.square(tgm), axis=2)
            
            plot_squared_error_tgm(tgm, path.parent / 'plots' / f'{file[:-4]}.png')

        except:
            print(f'Could not plot {file}')
            pass