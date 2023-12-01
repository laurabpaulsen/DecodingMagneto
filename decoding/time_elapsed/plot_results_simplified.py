import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

colours = ['#9ad8b1', "#9dbde6", '#5DBB63FF']

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


def plot_tgm_ax(tgm, ax, cbar_label='MSE', min_val=None, max_val=None, cmap="RdBu_r", colourbar = True):

    # plot the results
    if min_val is not None and max_val is not None:
        im = ax.imshow(tgm, origin='lower', cmap=cmap, vmin=min_val, vmax=max_val)
    else:
        im = ax.imshow(tgm, origin='lower', cmap=cmap)

    # add colorbar
    if colourbar:
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.05, shrink=0.5)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", size=10)
        cbar.ax.ticklabel_format(useOffset=False, style = 'plain')

    ax.set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

    return ax

def determine_row_col(params):
    if params['task'] == 'memory':
        row = 0
    elif params['task'] == 'visual':
        row = 1

    if params['predict'] == 'session day':
        col = 0
    elif params['predict'] == 'trial number':
        col = 1

    return row, col

def flatten_remove_nans(tgm):
    # flatten the tgm
    tgm = tgm.flatten()

    # remove nans
    tgm = tgm[~np.isnan(tgm)]

    return tgm



def plot_tgm_session_day(tgm_dict, measurement="MSE", save_path=None, trial_type = "combined", cmap="RdBu_r"):
    if measurement not in ["MSE", "correlation"]:
        raise ValueError("measurement must be either MSE or correlation")
    
    # set up gridspec
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.5, width_ratios=[1, 1, 0.2], height_ratios=[1, 1])



    # make the other axes
    axes = []
    for i in range(2):
        for j in range(2):
            fig.add_subplot(gs[i, j])
            axes.append(plt.gca())
            
            
   
    #combine the two rows in the last column
    ax_cbar = fig.add_subplot(gs[:, -1])
  
    # only keep combined trial type
    tgm_dict = {k: v for k, v in tgm_dict.items() if v["params"]["trial_type"] == trial_type}

    for key, value in tgm_dict.items():
        params = value["params"]
        tgm = value["tgm"]


        if params["task"] == "visual":
            if params["predict"] == "session day":
                ax_im = axes[0]
            elif params["predict"] == "session number":
                ax_im = axes[1]
        elif params["task"] == "memory":
            if params["predict"] == "session day":
                ax_im = axes[2]
            elif params["predict"] == "session number":
                ax_im = axes[3]
        else:
            continue

        plot_tgm_ax(tgm, ax=ax_im, cbar_label='Correlation', min_val=-1, max_val=1, cmap=cmap, colourbar=False)

    # add titles to the axes
    axes[0].set_title('Session day')
    axes[0].set_ylabel("Visual")
    axes[1].set_title('Session number')
    axes[2].set_ylabel('Memory')


    # Add colorbar to the colorbar axis
    im = ax_cbar.imshow(np.zeros((250, 250)), origin='lower', cmap=cmap, vmin=-1, vmax=1)
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom", size=10)

    

    # save the figure
    if save_path:
        plt.savefig(save_path / f'time_elapsed_{measurement}_{trial_type}.png', bbox_inches='tight')

def plot_diagonals(tgm_dict, measurement = "MSE", save_path = None, trial_type ="combined"):
    # figure with subplots
    fig, ax = plt.subplots(1, 1, figsize=(9, 7), sharey=True)
    
    # only keep combined trial type
    tgm_dict = {k: v for k, v in tgm_dict.items() if v["params"]["trial_type"] == trial_type}
    for key, value in tgm_dict.items():


        params = value["params"]
        tgm = value["tgm"]

        diagonal_values = tgm.diagonal()
        if params["predict"] == "trial number":
            continue
        if params['task'] == 'memory':
            colour = colours[0]
            if params['predict'] == 'session day':
                line_style = '-'
            elif params['predict'] == 'session number':
                line_style = '--'
        elif params['task'] == 'visual':
            colour = colours[1]
            if params['predict'] == 'session day':
                line_style = '-'
            elif params['predict'] == 'session number':
                line_style = '--'
        else:
            continue

        # plot the results
        ax.plot(diagonal_values, colour, linestyle=line_style, label=f'{params["task"]} {params["predict"]}')

        ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        ax.set_xlim(0, 250)

        # add a horizontal line at 0
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

        # add a title to the figure
        fig.supylabel(measurement.upper())
        fig.supxlabel('Time (s)'.upper())

        # place legend on the top axis
        ax.legend(loc='upper right')

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
    tmp = params.copy()
    tmp['trial_type'] = trial_type

    return tmp




def prepare_dicts(file_dict, path):
    MSE_dict = {}
    correlation_dict = {}

    for f, params in tqdm(file_dict.items(), desc="Preparing dictionaries with correlation and MSE results"):
        animate_file, inanimate_file, animate_true_file, inanimate_true_file = return_file_paths(path, f)

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
    path = Path(__file__).parent

    save_path = path / 'plots'

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

    
    dict_path = path / "dicts"
    
    if not dict_path.exists():
        dict_path.mkdir()

    # takes a while to calculate, so save the results
    if (dict_path / "MSE_dict.npy").exists() and (dict_path / "correlation_dict.npy").exists():
        print("Loading MSE and correlation dictionaries from file")
        MSE_dict = np.load(dict_path / "MSE_dict.npy", allow_pickle=True).item()
        correlation_dict = np.load(dict_path / "correlation_dict.npy", allow_pickle=True).item()
    else:
        MSE_dict, correlation_dict = prepare_dicts(tgm_files, path)

        np.save(dict_path / "MSE_dict.npy", MSE_dict)
        np.save(dict_path / "correlation_dict.npy", correlation_dict)
    
    
    measurement = "correlation"
    plot_tgm_session_day(correlation_dict, measurement=measurement, save_path=save_path, trial_type="animate", cmap = "RdBu_r")
    plot_diagonals(correlation_dict, measurement=measurement, save_path=save_path, trial_type="animate")

    #measurement = "MSE"    
    #plot_tgm_session_day(MSE_dict, measurement=measurement, save_path=save_path, trial_type="animate", cmap = "PuOr_r")