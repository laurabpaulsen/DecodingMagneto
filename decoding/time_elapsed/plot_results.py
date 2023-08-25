import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.ticker import FuncFormatter


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


def plot_tgm_ax(tgm, ax, cbar_label='MSE'):

    # plot the results
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


def plot_tgm_correlations(tgm_dict):
    for trial_type in ['animate', 'inanimate']:
        # figure with subplots
        fig, axs = plt.subplots(3, 3, figsize=(12, 9))
        
        for i, (file, params) in enumerate(tgm_dict.items()):
            if params['trial_type'] == trial_type:
                predicted = np.load(path.parent / 'results' / file, allow_pickle=True)
                true = np.load(path.parent / 'results' / file.replace('predict', 'true'), allow_pickle=True)

                if params['trial_type'] == trial_type:
                    # get the correlation between the predicted and true values for each timepoint
                    cor_tgm = np.zeros((250, 250))
                    for i in range(250):
                        for j in range(250):
                            # take only the non-nan values
                            tmp_predicted = predicted[i, j, :, :].flatten()
                            tmp_true = true[i, j, :, :].flatten()

                            # remove nans
                            tmp_predicted = tmp_predicted[~np.isnan(tmp_predicted)]
                            tmp_true = tmp_true[~np.isnan(tmp_true)]

                            # calculate the correlation
                            cor_tgm[i, j] = np.corrcoef(tmp_predicted, tmp_true)[0, 1]
                        

                    # based on the params determine the row and column of the subplot
                    row, col = determine_row_col(params)
                    
                    # plot the results
                    plot_tgm_ax(cor_tgm, axs[row, col], cbar_label='Correlation')


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
        plt.savefig(path.parent / 'plots' /f'time_elapsed_tgm_{trial_type}_corr.png', bbox_inches='tight')

def plot_tgm_MSE(tgm_dict):
    for trial_type in ['animate', 'inanimate']:

        fig, axs = plt.subplots(3, 3, figsize=(12, 9))

        for i, (file, params) in enumerate(tgm_dict.items()):
            if params['trial_type'] == trial_type:
                predicted = np.load(path.parent / 'results' / file, allow_pickle=True)
                true = np.load(path.parent / 'results' / file.replace('predict', 'true'), allow_pickle=True)
                        
                # get the MSE between the predicted and true values for each timepoint
                MSE_tgm = np.zeros((250, 250))
                for i in range(250):
                    for j in range(250):
                        # calculate the MSE between the predicted and true values across all stratification groups and trials
                        tmp_predicted = predicted[i, j, :, :].flatten()
                        tmp_true = true[i, j, :, :].flatten()

                        # remove nan values
                        tmp_predicted = tmp_predicted[~np.isnan(tmp_predicted)]
                        tmp_true = tmp_true[~np.isnan(tmp_true)]

                        # calculate the mean squared error
                        MSE_tgm[i, j] = np.mean((tmp_predicted - tmp_true)**2)


                # based on the params determine the row and column of the subplot
                row, col = determine_row_col(params)
                    
                # plot the results
                plot_tgm_ax(MSE_tgm, axs[row, col])


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
        plt.savefig(path.parent / 'plots' /f'time_elapsed_tgm_{trial_type}_mse.png', bbox_inches='tight')



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

    # plot_tgm_correlations(tgm_files)
    plot_tgm_MSE(tgm_files)

    # plot correlation between predicted and true values
    plot_tgm_correlations(tgm_files)