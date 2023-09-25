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

def plot_tgm(tgm_dict, measurement = "MSE", save_path = None, trial_type = None):
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))

    for key, value in tgm_dict.items():
        params = value["params"]
        tgm = value["tgm"]


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

        # tight layout
        fig.tight_layout()

        # save the figure
        if save_path:
            plt.savefig(save_path / f'time_elapsed_tgm_{measurement}_{trial_type}.png', bbox_inches='tight')


def plot_diagonals(tgm_dict, measurement = "MSE", save_path = None, trial_type = None):
    # figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharey=True)
    for key, value in tgm_dict.items():
        params = value["params"]
        tgm = value["tgm"]

        diagonal_values = tgm.diagonal()

        # based on the params determine the row and column of the subplot
        row, col = determine_row_col(params)

        # plot the results
        axs[row].plot(diagonal_values, label=params['predict'].upper(), color=colours[col])

        # add titles to the rows
        axs[0].set_title('Combined'.upper())
        axs[1].set_title('Memory'.upper())
        axs[2].set_title('Visual'.upper())

        for ax in axs:
            ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
            ax.legend(loc='upper right')
            ax.set_xlim(0, 250)

        # add a title to the figure
        fig.supylabel(measurement.upper())
        fig.supxlabel('Time (s)'.upper())

        # tight layout
        fig.tight_layout()

        # save the figure
        if save_path:
            plt.savefig(save_path /f'time_elapsed_diagonal_{measurement}_{trial_type}.png', bbox_inches='tight')

def return_file_paths(path, file):
    """
    Returns file paths for animate and inanimate versions of the same file, both the predicted and true values
    """

    animate_file = path / 'results' / file
    inanimate_file = path / 'results' / file.replace('animate', 'inanimate')

    animate_true_file = path / 'results' / file.replace('predict', 'true')
    inanimate_true_file = path / 'results' / file.replace('animate', 'inanimate').replace('predict', 'true')

    return animate_file, inanimate_file, animate_true_file, inanimate_true_file
    

def update_params(params, trial_type):
    """
    Updates the params dictionary to reflect the trial type
    """
    params["trial_type"] = trial_type

    return params

def prepare_dicts(file_dict, path):
    MSE_dict = {}
    correlation_dict = {}

    for f, params in file_dict.items():
        animate_file, inanimate_file, animate_true_file, inanimate_true_file = return_file_paths(path, f)
        print(animate_file, inanimate_file, animate_true_file, inanimate_true_file)

        predicted_animate = np.load(animate_file, allow_pickle=True)
        true_animate = np.load(animate_true_file, allow_pickle=True)

        # to get the inanimate results, we need to swap the animate and inanimate labels
        predicted_inanimate = np.load(inanimate_file, allow_pickle=True)
        true_inanimate = np.load(inanimate_true_file, allow_pickle=True)

        # combine the animate and inanimate results by concatenating them along the last axis
        predicted_combined = np.concatenate((predicted_animate.copy(), predicted_inanimate.copy()), axis=-1)
        true_combined = np.concatenate((true_animate.copy(), true_inanimate.copy()), axis=-1)

        for trial_type, (predicted, true) in zip(["animate", "inanimate", "combined"], [(predicted_animate, true_animate), (predicted_inanimate, true_inanimate), (predicted_combined, true_combined)]):
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

            # change params to reflect the trial type and the file name
            tmp_params = update_params(params, trial_type)
            f_tmp = f.replace("animate", trial_type)

            # save the tgm
            MSE_dict[f_tmp] = {"tgm": MSE_tgm, "params": tmp_params}
            correlation_dict[f_tmp] = {"tgm": correlation_tgm, "params": tmp_params}

    return MSE_dict, correlation_dict



if __name__ == "__main__":
    path = Path(__file__)

    save_path = path.parent / 'plots'

    # ensure that path to save plots exists
    if not save_path.exists():
        save_path.mkdir()

    tgm_files = {
        "animate_combined_predict_session_number.npy": {"predict": "session number", "task": "combined", "trial_type": "animate"},
        "animate_combined_predict_session_day.npy": {"predict": "session day", "task": "combined", "trial_type": "animate"},
        "animate_memory_predict_session_number.npy": {"predict": "session number", "task": "memory", "trial_type": "animate"},
        "animate_memory_predict_session_day.npy": {"predict": "session day", "task": "memory", "trial_type": "animate"},
        "animate_visual_predict_session_number.npy": {"predict": "session number", "task": "visual", "trial_type": "animate"},
        "animate_visual_predict_session_day.npy": {"predict": "session day", "task": "visual", "trial_type": "animate"},
        "animate_combined_predict_trial_number.npy": {"predict": "trial number", "task": "combined", "trial_type": "animate"},
        "animate_memory_predict_trial_number.npy": {"predict": "trial number", "task": "memory", "trial_type": "animate"},
        "animate_visual_predict_trial_number.npy": {"predict": "trial number", "task": "visual", "trial_type": "animate"},
        }

    MSE_dict, correlation_dict = prepare_dicts(tgm_files, path.parent)

    for trial_type in ["animate", "inanimate", "combined"]:
        print (f"Working on {trial_type}")
        tmp_dict = {}
        for key, value in MSE_dict.items():
            if value["params"]["trial_type"] == trial_type:
                tmp_dict[key] = value
                print(key)

        # plot the results for each trial type
        plot_tgm(tmp_dict, measurement="MSE", save_path=save_path, trial_type=trial_type)
        plot_diagonals(tmp_dict, measurement="MSE", save_path=save_path, trial_type=trial_type)



    # plot the results
    for measurement, dictionary in zip(["correlation", "MSE"], [correlation_dict, MSE_dict]):
        plot_tgm(dictionary, measurement=measurement)
        plot_diagonals(dictionary, measurement=measurement)
