import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

colours = ['#0063B2FF', '#5DBB63FF', 'lightblue']

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


def plot_tgm_ax(tgm, ax, cbar_label='MSE', min_val=None, max_val=None):

    # plot the results
    if min_val is not None and max_val is not None:
        im = ax.imshow(tgm, origin='lower', cmap="autumn_r", vmin=min_val, vmax=max_val)
    else:

        im = ax.imshow(tgm, origin='lower', cmap="autumn_r")

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.01, shrink=0.8)
    
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", size=10)
    cbar.ax.ticklabel_format(useOffset=False, style = 'plain')



    ax.set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

    return ax

def determine_row_col(params):
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
    
    return row, col

def flatten_remove_nans(tgm):
    # flatten the tgm
    tgm = tgm.flatten()

    # remove nans
    tgm = tgm[~np.isnan(tgm)]

    return tgm


def plot_tgm(tgm_dict, measurement = "MSE"):
    for trial_type in ['animate', 'inanimate']:

        fig, axs = plt.subplots(3, 3, figsize=(12, 9))

        for key, value in tgm_dict.items():
            params = value["params"]
            tgm = value["tgm"]

            if params['trial_type'] == trial_type:                        
                # based on the params determine the row and column of the subplot
                row, col = determine_row_col(params)

                # plot the results
                if measurement == "MSE":
                    plot_tgm_ax(tgm, axs[row, col])
                elif measurement == "correlation":
                    plot_tgm_ax(tgm, axs[row, col], cbar_label='Correlation', min_val=-1, max_val=1)


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

        # tight layout
        fig.tight_layout()
        
        # save the figure
        plt.savefig(path.parent / 'plots' /f'time_elapsed_tgm_{trial_type}_{measurement}.png', bbox_inches='tight')


def plot_diagonals(tgm_dict, measurement = "MSE"):
    for trial_type in ['animate', 'inanimate']:
        # figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharey=True)
        for key, value in tgm_dict.items():
            params = value["params"]
            tgm = value["tgm"]
            
            if params['trial_type'] == trial_type:
                # get the correlation between the predicted and true values for each timepoint
                diagonal_values = tgm.diagonal()
                        
                # based on the params determine the row and column of the subplot
                row, col = determine_row_col(params)
                    
                # plot the results
                axs[row].plot(diagonal_values, label=params['predict'], color=colours[col])

        # add titles to the rows
        axs[0].set_title('Combined'.upper())
        axs[1].set_title('Memory'.upper())
        axs[2].set_title('Visual'.upper())

        for ax in axs:
            ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
            ax.legend(loc='upper right')
            ax.set_xlim(0, 250)

        # add a title to the figure
        fig.suptitle(f'{trial_type.capitalize()} Diagonals'.upper())
        fig.supylabel(measurement.upper())
        fig.supxlabel('Time (s)'.upper())

        # tight layout
        fig.tight_layout()
        
        # save the figure
        plt.savefig(path.parent / 'plots' /f'time_elapsed_diagonal_{trial_type}_{measurement}.png', bbox_inches='tight')



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
        "inanimate_visual_predict_session_day.npy": {"predict": "session day", "task": "visual", "trial_type": "inanimate"},
        "animate_combined_predict_trial_number.npy": {"predict": "trial number", "task": "combined", "trial_type": "animate"},
        "inanimate_combined_predict_trial_number.npy": {"predict": "trial number", "task": "combined", "trial_type": "inanimate"},
        "animate_memory_predict_trial_number.npy": {"predict": "trial number", "task": "memory", "trial_type": "animate"},
        "inanimate_memory_predict_trial_number.npy": {"predict": "trial number", "task": "memory", "trial_type": "inanimate"},
        "animate_visual_predict_trial_number.npy": {"predict": "trial number", "task": "visual", "trial_type": "animate"},
        "inanimate_visual_predict_trial_number.npy": {"predict": "trial number", "task": "visual", "trial_type": "inanimate"},
    }

    
    MSE_dict = {}
    correlation_dict = {}

    for f, params in tgm_files.items():
        # load the predicted and true values
        predicted = np.load(path.parent / 'results' / f, allow_pickle=True)
        true = np.load(path.parent / 'results' / f.replace('predict', 'true'), allow_pickle=True)

        # get the MSE and correlation between the predicted and true values for each timepoint
        MSE_tgm = np.zeros((250, 250))
        correlation_tgm = np.zeros((250, 250))

        for i in range(250):
            for j in range(250):
                # take only the non-nan values and flatten the array
                tmp_predicted = flatten_remove_nans(predicted[i, j, :, :])
                tmp_true = flatten_remove_nans(true[i, j, :, :])
            
                # calculate the mean squared error
                MSE_tgm[i, j] = np.mean((tmp_predicted - tmp_true)**2)
                correlation_tgm[i, j] = np.corrcoef(tmp_predicted, tmp_true)[0, 1]
        
        # save the tgm
        MSE_dict[f] = {"tgm": MSE_tgm, "params": params}
        correlation_dict[f] = {"tgm": correlation_tgm, "params": params}

    # plot the results
    for measurement, dictionary in zip(["correlation", "MSE"], [correlation_dict, MSE_dict]):
        plot_tgm(dictionary, measurement=measurement)
        plot_diagonals(dictionary, measurement=measurement)
