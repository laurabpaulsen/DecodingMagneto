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


def plot_squared_error_tgm_ax(tgm, ax):

    # plot the results
    im = ax.imshow(tgm, origin='lower', cmap="autumn_r")

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # add title to colorbar
    cbar.set_label('MSE', size=10)

    ax.set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

    return ax



if __name__ == "__main__":
    path = Path(__file__)
    
    tgm_files = {
        "animate_combined_predict_session_number.npy": {"predict": "session number", "task": "combined", "trial_type": "animate"},
        "animate_combined_predict_session_day.npy": {"predict": "session day", "task": "combined", "trial_type": "animate"},
        "inanimate_combined_predict_session_number.npy": {"predict": "session number", "task": "combined", "trial_type": "inanimate"},
        "inanimate_combined_predict_session_day.npy": {"predict": "session day", "task": "combined", "trial_type": "inanimate"},
        "animate_memory_predict_session_number.npy": {"predict": "session number", "task": "memory", "trial_type": "animate"},
        "animate_memory_predict_session_day.npy": {"predict": "session day", "task": "memory", "trial_type": "animate"},
        "inanimate_memory_predict_session_number.npy": {"predict": "session number", "task": "memory", "trial_type": "inanimate"},
        "inanimate_memory_predict_session_day.npy": {"predict": "session day", "task": "memory", "trial_type": "inanimate"},
        "animate_visual_predict_session_number.npy": {"predict": "session number", "task": "visual", "trial_type": "animate"},
        "animate_visual_predict_session_day.npy": {"predict": "session day", "task": "visual", "trial_type": "animate"},
        "inanimate_visual_predict_session_number.npy": {"predict": "session number", "task": "visual", "trial_type": "inanimate"},
        "inanimate_visual_predict_session_day.npy": {"predict": "session day", "task": "visual", "trial_type": "inanimate"}
    }

    for trial_type in ['animate', 'inanimate']:
        # figure with subplots
        fig, axs = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
        
        for i, (file, params) in enumerate(tgm_files.items()):
            if params['trial_type'] == trial_type:
                try:
                    tgm = np.load(path.parent / 'results' / file, allow_pickle=True)
                    
                    # calculate the mean squared error across the 2 axis (contains the error)
                    tgm = np.mean(np.square(tgm), axis=2)

                    # based on the params determine the row and column of the subplot
                    if params['task'] == 'combined':
                        row = 0
                    elif params['task'] == 'memory':
                        row = 1
                    elif params['task'] == 'visual':
                        row = 2
                    
                    if params['predict'] == 'session number':
                        col = 0
                    elif params['predict'] == 'session day':
                        col = 1
                    elif params['predict'] == 'trial number':
                        col = 2
                
                    # plot the results
                    plot_squared_error_tgm_ax(tgm, axs[row, col])

                except:
                    print(f'Could not plot {file}')
                    pass

        # add titles to the columns
        axs[0, 0].set_title('Session Number'.upper())
        axs[0, 1].set_title('Session Day'.upper())
        axs[0, 2].set_title('Trial Number'.upper())

        # add titles to the rows
        axs[0, 0].set_ylabel('Combined'.upper())
        axs[1, 0].set_ylabel('Memory'.upper())
        axs[2, 0].set_ylabel('Visual'.upper())

        # add a title to the figure
        fig.suptitle(f'{trial_type.capitalize()} TGMs'.upper())
        
        # save the figure
        plt.savefig(path.parent / 'plots' /f'time_elapsed_tgm_{trial_type}.png', bbox_inches='tight')