import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pickle as pkl


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

    ax.set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ], size = 7)
    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ], size = 7)

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





def return_file_paths(path, file):
    """
    Returns file paths for animate and inanimate versions of the same file, both the predicted and true values
    """

    animate_file = path / 'results' / file
    inanimate_file = path / 'results' / file.replace('animate', 'inanimate')

    return animate_file, inanimate_file, 

    

def update_params(params, trial_type):
    """
    Updates the params dictionary to reflect the trial type
    """
    tmp = params.copy()
    tmp['trial_type'] = trial_type

    return tmp


def get_correlation_tgm(predicted, true):
    """
    Calculate the correlation between the predicted and true values for each timepoint
    """
    correlation_tgm = np.zeros((250, 250))
    for i in range(250):
        for j in range(250):
            # take only the non-nan values and flatten the array
            tmp_predicted = flatten_remove_nans(predicted[i, j, :, :])
            tmp_true = flatten_remove_nans(true[i, j, :, :])

            # calculate the correlation
            correlation_tgm[i, j] = np.corrcoef(tmp_predicted, tmp_true)[0, 1]

    return correlation_tgm


def prepare_dicts(file_dict, path):

    output_all = {}
    for f, params in tqdm(file_dict.items(), desc="Preparing dictionaries with correlation results"):
        animate_file, inanimate_file = return_file_paths(path, f)

        animate = pkl.load(open(animate_file, "rb"))
        inanimate = pkl.load(open(inanimate_file, "rb"))

        for trial_type, results in zip(["animate", "inanimate"], [animate, inanimate]):

            tmp_params = update_params(params, trial_type)
            f_tmp = f.replace("animate", trial_type)
            
            # loop over the results and permuted results
            output_tmp = {}
            for key, value in results.items():
                if key == "original":
                    permutation = "original"

                else:
                    permutation = value["correlation"]


                # get correlation between the predicted and true values for each timepoint
                correlation_tgm = get_correlation_tgm(value["predicted"], value["true"])

                # add to the output dictionary
                output_tmp[key] = {
                    "params": tmp_params,
                    "corr_permutation_y_true_y": permutation,
                    "tgm": correlation_tgm
                }

            output_all[f_tmp] = output_tmp

    return output_all



def plot_results(tgm_dict, save_path=None, trial_type="animate", session_type="memory", cmap="RdBu_r", predict="session number", min_val=-1, max_val=1):    
    tmp_dict = {}

    for key, value in tgm_dict.items():
        for key2, value2 in value.items():
            params = value2["params"]


            if params["trial_type"] == trial_type and params["predict"] == predict and params["task"] == session_type:
                tmp_dict[key2] = value2

    fig = plt.figure(figsize=(8, 12), dpi=300)
    gs = plt.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.5, width_ratios=[1, 1, 1], height_ratios=[2, 1, 1, 1])


     # the two top rows are used for the unpermuted results
    ax_original = fig.add_subplot(gs[:2, :])

    # the two last rows are used for the permuted results
    ax_permuted = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])]

    # plot the unpermuted results
    for key, value in tmp_dict.items():
        params = value["params"]
        tgm = value["tgm"]

        if type(key) == str:
            ax = ax_original
            ax.set_title(key.capitalize())
            colourbar = True
        else:
            ax = ax_permuted.pop(0)
            ax.set_title(f"Permution {key}", fontsize=10)

            colour = "grey"
            colourbar = False
            
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
            

            ax.tick_params(axis='both', colors=colour)

            # set a label with the correlation between y_true and y_permutation
            ax.text(
                1, 1, 
                f"Distance permuted order and right order \n {value['corr_permutation_y_true_y']:.2f}",
                ha="right", va="top", fontsize=5, color = "k", transform=ax.transAxes
                )

        plot_tgm_ax(tgm, ax=ax, cbar_label=f"Correlation \n predicted {predict} and true values", min_val=min_val, max_val=max_val, cmap=cmap, colourbar=colourbar)



    if save_path:
        plt.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
    path = Path(__file__).parent

    save_path = path / 'plots'

    # ensure that path to save plots exists
    if not save_path.exists():
        save_path.mkdir()

    
    tgm_files = {
        "animate_memory_session_number.pkl": {"predict": "session number", "task": "memory", "trial_type": "animate"},
        "animate_memory_session_day.pkl": {"predict": "session day", "task": "memory", "trial_type": "animate"},
        "animate_visual_session_number.pkl": {"predict": "session number", "task": "visual", "trial_type": "animate"},
        "animate_visual_session_day.pkl": {"predict": "session day", "task": "visual", "trial_type": "animate"},
        }

    
    dict_path = path / "dicts"
    
    if not dict_path.exists():
        dict_path.mkdir()

    # takes a while to calculate, so save the results
    if (dict_path / "correlation_dict.npy").exists():
        print("Loading correlation dictionaries from file")
        correlation_dict = np.load(dict_path / "correlation_dict.npy", allow_pickle=True).item()
    else:
        correlation_dict = prepare_dicts(tgm_files, path)
        np.save(dict_path / "correlation_dict.npy", correlation_dict)
    
    predict = "session day"

    for trial_type in ["animate", "inanimate"]:
        for session_type in ["memory", "visual"]:
            for predict in ["session day", "session number"]:
                plot_results(
                    correlation_dict, 
                    save_path = save_path / f'{trial_type}_{session_type}_{predict.replace(" ", "")}.png', 
                    trial_type = trial_type, 
                    session_type = session_type,
                    predict = predict
                    )
