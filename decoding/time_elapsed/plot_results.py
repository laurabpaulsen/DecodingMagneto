import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl
import pandas as pd
from mne.stats import permutation_cluster_test

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

    results = {}
   
    with open(results_path, 'rb') as fr:
        try:
            while True:
                tmp_data = pkl.load(fr)
                key = list(tmp_data.keys())[0]
                value = tmp_data[key]

                results[key] = value

        except EOFError:
            pass

    output = {}     
    # loop over the results and permuted results    
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

def plot_results(tgm_dict, save_path=None, cmap="RdBu_r", min_val=-1, max_val=1):    
    # figuring out size of figure and gridspec
    n_permutations = len(tgm_dict) - 1
    n_rows = 2 + n_permutations//3
    n_cols = 3
    
    fig = plt.figure(figsize=(n_cols*2.5 , n_rows*3), dpi=300)

    gs = plt.GridSpec(n_rows, n_cols, figure=fig, hspace=0.5, wspace=0.5, width_ratios=[1, 1, 1])

    # the two top rows are used for the unpermuted results
    ax_original = fig.add_subplot(gs[:2, :])

    # the two last rows are used for the permuted results

    ax_permuted = []
    for i in range(2, n_rows):
        for j in range(n_cols):
            ax_permuted.append(fig.add_subplot(gs[i, j]))
   
    # plot the unpermuted results
    for key, value in tgm_dict.items():
        tgm = value["tgm"]

        if key == "original":
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

        plot_tgm_ax(tgm, ax=ax, cbar_label=f"Correlation \n predicted and true values", min_val=min_val, max_val=max_val, cmap=cmap, colourbar=colourbar)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()



def create_table(correlation_dict, timesample = 100):
    # create a table with the correlation between the true and permuted y values
    df = pd.DataFrame(columns=["permutation", "correlation_per_y_true_y", "corr_timesample", "timesample"])
    for key, value in correlation_dict.items():

        # get the correlation at the timesample
        corr_ts = value["tgm"][timesample, timesample]

        tmp = pd.DataFrame({"permutation": [key], "correlation_per_y_true_y": [value["corr_permutation_y_true_y"]], "corr_timesample": [corr_ts], "timesample": [timesample]})

        df = pd.concat([df, tmp], axis=0)


    return df

def plot_results_diagonals(tgm_dict, save_path=None, cmap="viridis_r", y_min = -0.2, y_max = 0.75):    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=300)

    cmap = plt.colormaps[cmap]
    for i, (key, value) in enumerate(tgm_dict.items()):
        tgm = value["tgm"]
        diagonal = np.diag(tgm)
        corr_y_true_y = value["corr_permutation_y_true_y"]

        if key == "original":
            ax.plot(diagonal, label=key.capitalize(), color="k")
        else:
            # colour by the correlation between y_true and y_permutation
            ax.plot(diagonal, color= "forestgreen", alpha=0.5, linewidth=0.5, label = f"Permuted")

    # legend (but only with one permuted label)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.legend(unique_handles, unique_labels)

    # make a permutation test to see if the correlation between the predicted and true values is significantly different from the permuted values
    # calculate p-value
    permuted = [np.diag(value["tgm"]) for key, value in tgm_dict.items() if type(key) != "original"]
    permutations_corr = np.array(permuted)

    # original
    original = np.diag(tgm_dict["original"]["tgm"]).reshape(1, -1)

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [original, permutations_corr],
        n_permutations=1000, 
        threshold=0.05)
    
    # plot significant clusters as stars at the bottom of the plot
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] < 0.05:
            ax.plot(c, np.ones_like(c)*-0.15, color="grey", linewidth=2, alpha = 0.5)

    # plot the colourbar
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    #sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax)

    # set the colourbar label
    #cbar.set_label("Correlation between true y and permuted y", rotation=-90, va="bottom")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Correlation (predicted and true)")

    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 250)

            
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    path = Path(__file__).parent

    save_path = path / 'plots'
    save_path_extra = save_path / 'extra'

    # make dirs for saving if they don't exist
    if not save_path.exists():
        save_path.mkdir()
    
    if not save_path_extra.exists():
        save_path_extra.mkdir()
        
    dict_path = path / "dicts"
    
    if not dict_path.exists():
        dict_path.mkdir()

    # looping over each trial type, session type and prediction type
    for trial_type in ["animate", "inanimate"]:
        for session_type in ["visual"]:
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
                    # take the 7 first entries in the dictionary (original and 6 permutations)
                    dict(list(correlation_dict.items())[:7]),
                    save_path = save_path / f'{trial_type}_{session_type}_{predict.replace(" ", "")}.png'
                    )
                
                plot_results(
                    correlation_dict,
                    save_path = save_path_extra / f'{trial_type}_{session_type}_{predict.replace(" ", "")}.png'
                    )
                
                plot_results_diagonals(
                    correlation_dict,
                    save_path = save_path_extra / f'diagonals_{trial_type}_{session_type}_{predict.replace(" ", "")}.png'
                    )
                
                # get the timesample for 200 ms
                hz = 250
                ts = int(0.2 * hz)
                table = create_table(correlation_dict, timesample = ts)

            
                table.to_csv(save_path_extra / f'{trial_type}_{session_type}_{predict.replace(" ", "")}_table.csv', index=False)
