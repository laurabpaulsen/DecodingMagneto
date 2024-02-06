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


def plot_tgm_ax(tgm, ax, cbar_label='MSE', min_val=None, max_val=None, cmap="RdBu_r", colourbar = False):

    # plot the results

    im = ax.imshow(tgm, origin='lower', cmap=cmap, vmin=min_val, vmax=max_val)

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



def plot_tgm_session_day(tgm_dict, measurement="MSE", save_path=None, trial_type = "animate", cmap="RdBu_r", predict = "session number", min_val=None, max_val=None):
    if measurement not in ["MSE", "correlation"]:
        raise ValueError("measurement must be either MSE or correlation")
    
    # set up gridspec
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(1, 4, hspace=0.2, wspace=0.5, width_ratios=[1, 1, 1, 0.2], height_ratios=[1])


    # make axes for each task
    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[i])
        axes.append(ax)
            
    # add colorbar axis
    ax_cbar = fig.add_subplot(gs[-1])
  
    # only keep the trial type we want (animate or inanimate)
    tgm_dict = {k: v for k, v in tgm_dict.items() if v["params"]["trial_type"] == trial_type}

    # only keep the predict we want (session day, session number, or trial number)
    tgm_dict = {k: v for k, v in tgm_dict.items() if v["params"]["predict"] == predict}

    for key, value in tgm_dict.items():
        params = value["params"]
        tgm = value["tgm"]


        if params["task"] == "visual":
            ax_im = axes[0]
            ax_im.set_title("Visual")

        elif params["task"] == "memory":
            ax_im = axes[1]
            ax_im.set_title("Memory")

        elif params["task"] == "combined":
            ax_im = axes[2]
            ax_im.set_title("Combined")

        else:
            continue

        plot_tgm_ax(tgm, ax=ax_im, cbar_label=measurement, min_val=min_val, max_val=max_val, cmap=cmap)

        # print the max value
        print(f'{key}: {tgm.max()}')


    # Add colorbar to the colorbar axis
    im = ax_cbar.imshow(np.zeros((250, 250)), origin='lower', cmap=cmap, vmin=min_val, vmax=max_val)
    cbar = fig.colorbar(im, cax=ax_cbar, fraction=2)
    cbar.ax.set_ylabel(measurement.title(), rotation=-90, va="bottom", size=10)

    # add more ticks to the colorbar
    if measurement == "correlation":
        cbar.ax.set_yticks(np.arange(-1, 1.1, step=0.5))


    # save the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')



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
    trial_type = "animate"
    predict = "session day"
    
    plot_tgm_session_day(
        correlation_dict, 
        measurement=measurement, 
        save_path=save_path / f'time_elapsed_{measurement}_{trial_type}_{predict.replace(" ", "")}.png',
        predict=predict,
        trial_type=trial_type, 
        min_val=-1,
        max_val=1,
        cmap = "RdBu_r")

    predict = "session number"
    plot_tgm_session_day(
        correlation_dict, 
        measurement=measurement, 
        predict=predict,
        save_path=save_path / f'time_elapsed_{measurement}_{trial_type}_{predict.replace(" ", "")}.png',
        trial_type=trial_type, 
        min_val=-1,
        max_val=1,
        cmap = "RdBu_r")

