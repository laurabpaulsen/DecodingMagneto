import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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


def flatten_remove_nans(tgm):
    # flatten the tgm
    tgm = tgm.flatten()

    # remove nans
    tgm = tgm[~np.isnan(tgm)]

    return tgm
    

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


def prepare_dicts(results_path):

    results = pkl.load(open(results_path, "rb"))

            
    # loop over the results and permuted results
    output = {}
    
    for key, value in results.items():
        if key == "original":
            permutation = "original"

        else:
            permutation = value["correlation"]


        # get correlation between the predicted and true values for each timepoint
        correlation_tgm = get_correlation_tgm(value["predicted"], value["true"])

        # add to the output dictionary
        output[key] = {
            "corr_permutation_y_true_y": permutation,
            "tgm": correlation_tgm
        }
    
    # reorder dictionary so original is first (helps with plotting)
    sorted_keys = sorted(output.keys(), key=lambda x: (x != 'original', x)) # sort the keys so that 'original' is first
    output = {key: output[key] for key in sorted_keys} # create a new dictionary with the sorted keys


    return output

def plot_results(tgm_dict, save_path=None, cmap="RdBu_r", predict="session number", min_val=-1, max_val=1):    
    fig = plt.figure(figsize=(8, 12), dpi=300)
    gs = plt.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.5, width_ratios=[1, 1, 1], height_ratios=[2, 1, 1, 1])

    # the two top rows are used for the unpermuted results
    ax_original = fig.add_subplot(gs[:2, :])

    # the two last rows are used for the permuted results
    ax_permuted = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])]

    # plot the unpermuted results
    for key, value in tgm_dict.items():
        tgm = value["tgm"]

        if type(key) == str:
            ax = ax_original
            ax.set_title(key.capitalize())
            colourbar = True
        else:
            try:
                ax = ax_permuted.pop(0)
            except IndexError: # if there are more permuations than axes available, break
                break
            ax.set_title(f"Permution {key}", fontsize=10)

            colour = "grey"
            colourbar = False
            
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
            
            ax.tick_params(axis='both', colors=colour)

            # set a label with the correlation between y_true and y_permutation
            ax.text(
                1.05, 1.05, 
                f"{value['corr_permutation_y_true_y']:.2f}",
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

        
    dict_path = path / "dicts"
    
    if not dict_path.exists():
        dict_path.mkdir()

    # looping over each trial type, session type and prediction type
    for trial_type in ["animate", "inanimate"]:
        for session_type in ["memory", "visual", "visualsubset"]:
            for predict in ["session day", "session number"]:

                file_path = dict_path / f"correlation_dict_{trial_type}_{session_type}_{predict.replace(' ', '')}.npy"

                if (file_path).exists():                           
                    print(f"Loading correlation dict for {trial_type} {session_type} {predict} from file")
                    correlation_dict = np.load(file_path, allow_pickle=True).item()

                else:
                    print(f"Calculating correlation dict for {trial_type} {session_type} {predict}")
                    results_path = path / "results" / f"{trial_type}_{session_type}_{predict.replace(' ', '_')}.pkl"
                    correlation_dict = prepare_dicts(results_path)
                    np.save(file_path, correlation_dict) # save the dictionary to file to be loaded in if script is run again

                plot_results(
                    correlation_dict, 
                    save_path = save_path / f'{trial_type}_{session_type}_{predict.replace(" ", "")}.png'
                    )
