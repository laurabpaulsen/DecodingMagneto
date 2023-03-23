"""
This script plots the results of the cross decoding analysis

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from utils.analysis import plot
from utils.analysis.tools import chance_level
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

alpha = 0.05

# set parameters for all plots
plt.rcParams['font.family'] = 'times new roman'
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

def determine_linestyle(train_condition, test_condition):
    if train_condition == test_condition:
        return "-"
    elif train_condition != test_condition:
        return "--"
    
def determine_colour(i, j):
    colors = sns.color_palette("Blues_r", 15)

    return colors[abs(i-j)]

def add_diagonal_line(ax):
    """
    Adds a diagonal line to a plot
    """
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = 'red', linestyle = '--', alpha = 0.5, linewidth = 1)

def change_spine_colour(ax, colour):
    """
    Changes the colour of the spines of a plot
    """
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)

    # change the colour of the ticks
    ax.tick_params(axis='x', colors=colour)
    ax.tick_params(axis='y', colors=colour)

    # title
    ax.title.set_color(colour)


def plot_cross_decoding_matrix(acc, parc):
    fig, axs = plt.subplots(acc.shape[0], acc.shape[1], figsize = (30, 30))
    
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            axs[i, j] = plot.plot_tgm_ax(acc[i, j], ax=axs[i, j], vmin=35, vmax=65)
            axs[i, j].set_title(f'train:{i+1}, test:{j+1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'cross_decoding_{parc}.png'))
    plt.close()


def plot_train_test_condition(acc, parc, vmin = 40, vmax = 60, diff_colour = 'darkblue'):
    fig, axs = plt.subplots(4, 3, figsize = (15, 15), dpi = 300)
    
    vis = np.array([0, 1, 2, 3, 4, 5, 6])
    mem = np.array([7, 8, 9, 10])
    
    n_trials_vis = 588 * len(vis)
    n_trials_mem = 588 * len(mem)

    vmin_diff = -5
    vmax_diff = 5
    
    # train and test on same condition
    vis_vis = np.nanmean(acc[vis,:, :, :][:, vis, :, :], axis = (0, 1))
    axs[1, 0] = plot.plot_tgm_ax(vis_vis, ax=axs[1, 0], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_vis, alpha = alpha, p = 0.5))
    axs[1, 0].set_title('train: vis,  test:vis')

    mem_mem = np.nanmean(acc[mem,:, :, :][:, mem, :, :], axis = (0, 1))

    axs[1, 1] = plot.plot_tgm_ax(mem_mem, ax=axs[1, 1], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_mem, alpha = alpha, p = 0.5))
    axs[1, 1].set_title('train:mem, test:mem')

    # difference between test and train condition (vis - mem)
    axs[1, 2] = plot.plot_tgm_ax(vis_vis - mem_mem, ax=axs[1, 2], vmin=vmin_diff, vmax=vmax_diff)
    
    axs[1, 2].set_title('vis_vis - mem_mem')

    # train on vis, test on mem
    vis_mem = acc[vis,:, :, :][:, mem, :, :].mean(axis = (0, 1))

    axs[2, 0] = plot.plot_tgm_ax(vis_mem, ax=axs[2, 0], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_mem,alpha = alpha, p = 0.5))
    axs[2, 0].set_title('train:vis, test:mem')

    # train on mem, test on vis
    mem_vis = acc[mem, :, :, :][:, vis, :, :].mean(axis = (0, 1))
    axs[2, 1] = plot.plot_tgm_ax(mem_vis, ax=axs[2, 1], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_vis, alpha = alpha, p = 0.5))
    axs[2, 1].set_title('train:mem, test:vis')

    # difference between test and train condition
    axs[2, 2] = plot.plot_tgm_ax(vis_mem - mem_vis, ax=axs[2, 2], vmin=vmin_diff, vmax=vmax_diff)
    axs[2, 2].set_title('vis_mem - mem_vis')

    # difference between vis_vis and vis_mem
    axs[3, 0] = plot.plot_tgm_ax(vis_vis - vis_mem, ax=axs[3, 0], vmin=vmin_diff, vmax=vmax_diff)
    axs[3, 0].set_title('vis_vis - vis_mem')
    
    # difference between mem_mem and mem_vis
    axs[3, 1] = plot.plot_tgm_ax(mem_mem - mem_vis, ax=axs[3, 1], vmin=vmin_diff, vmax=vmax_diff)
    axs[3, 1].set_title('mem_mem - mem_vis')

    # difference between vis_vis and mem_vis
    axs[3, 2] = plot.plot_tgm_ax(vis_vis - mem_vis, ax=axs[3, 2], vmin=vmin_diff, vmax=vmax_diff)
    axs[3, 2].set_title('vis_vis - mem_vis')

    # difference between vis_mem and mem_mem
    axs[0, 2] = plot.plot_tgm_ax(vis_mem - mem_mem, ax=axs[0, 2], vmin=vmin_diff, vmax=vmax_diff)
    axs[0, 2].set_title('vis_mem - mem_mem')

    for ax in axs[[2, 3, 3, 3, 1, 3, 0], [2, 2, 0, 1, 2, 2, 2]].flatten(): # difference plots
        change_spine_colour(ax, diff_colour)
        add_diagonal_line(ax)

    # plot colourbars in the first two columns of the first row

    gs = axs[0, 0].get_gridspec()

    # remove the underlying axes
    for ax in axs[0, :2]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, :2])
    axbig.axis('off')
    divider = make_axes_locatable(axbig)

    cax = divider.append_axes("top", size="20%", pad=0.1)
    cax1 = divider.append_axes("bottom", size="20%", pad=0.1)
    clb1 = fig.colorbar(axs[1, 0].images[0], cax = cax, orientation='horizontal', shrink = 2, pad=10)
    clb1.ax.set_title('accuracy (%)')

    clb2 = fig.colorbar(axs[1, 2].images[0], cax = cax1, orientation='horizontal', shrink = 2, pad=10)
    clb2.ax.set_title('difference (%)')
    change_spine_colour(clb2.ax, diff_colour)

    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'cross_decoding_{parc}_average_vis_mem.png'))
    plt.close()

def plot_diagonals(acc_dict, title = 'diagonals', save_path = None):
    """
    
    """

    dict_diag = {}
    for parc, acc in acc_dict.items():
        # check if any nans
        if np.isnan(acc).any():
            avg = np.nanmean(np.nanmean(acc, axis=0), axis=0)
            # take the diagonal
            dict_diag[parc] = np.diag(avg)

        else:
            within_session = np.diagonal(acc, axis1=0, axis2=1)
            # take the mean over sessions
            avg = np.nanmean(within_session, axis=2)
            dict_diag[parc] = np.diag(avg)


    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for key, value in dict_diag.items():
        ax.plot(value, label=key)

    ax.set_xlabel('time (ms)')
    ax.set_ylabel('accuracy')
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path)


def cross_diags_per_sesh(accuracies, save_path = None):
    diagonals = np.diagonal(accuracies, axis1=2, axis2=3)

    condition = ["vis", "vis", "vis", "vis", "vis", "vis", "vis", "mem", "mem", "mem", "mem"]
    
    fig, axs = plt.subplots(4, 3, figsize = (30, 25), dpi = 300, sharex = True, sharey = True)

    for i, ax in enumerate(axs.flatten()):

        if ax != axs.flatten()[-1]: 
            # title
            ax.set_title(f'Training on session {i+1}')
            
            # plot the diagonal
            for j in range(11):
                # determine line type
                line_style = determine_linestyle(condition[i], condition[j])

                # determine colour (depending on how far away the train sesh and test sesh are)
                col = determine_colour(i, j)
                    
                ax.plot(diagonals[i, j], color = col, alpha = 1, linewidth = 1, linestyle = line_style)

        # legend in last ax
        else:
            ax.axis('off')
            # add info to the legend
            for j in range(11):
                # determine colour (depending on how far away the train sesh and test sesh are)
                col = determine_colour(0, j)
                ax.plot([], [], color = col, alpha = 1,  linewidth = 1, label = f'{j}')
            
            ax.legend(title = 'Distance between training and test session',  ncol=len(condition), loc = 'center', bbox_to_anchor = (0.5, 0.5))
    # add labels
    fig.supylabel('Accuracy (%)')
    fig.supxlabel('Samples')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

def cross_diags_average_sesh(accuracies, save_path = None):
    diagonals = np.diagonal(accuracies, axis1=2, axis2=3)

    # create 11 by 11 matrix with distances between sessions
    # empty matrix
    distances = np.zeros((11, 11))

    # fill in the matrix
    for i in range(11):
        for j in range(11):
            distances[i, j] = abs(i-j)
    fig, ax = plt.subplots(1, 1, figsize = (20, 15), dpi = 300)

    # loop over all distances
    for dist in range(11):
        # skip distance 0
        if dist == 0:
            pass
        else:
            # get the indices of the sessions that are dist away from each other
            indices = np.argwhere(distances == dist)

            # get the average accuracy for each pair of sessions that are dist away from each other
            tmp = np.array([diagonals[i, j] for i, j in indices])

            # standard deviation
            tmp_std = tmp.std(axis = 0)
            tmp_acc = tmp.mean(axis = 0)

            colour = determine_colour(12, dist)
            
            # plot tmp acc
            ax.plot(tmp_acc, label = f'{dist} sessions apart', linewidth = 1, color = colour)

            # standard error
            tmp_std = tmp_std / np.sqrt(tmp.shape[0])

            # plot standard error
            ax.fill_between(np.arange(0, 250), (tmp_acc - tmp_std), (tmp_acc + tmp_std), alpha = 0.1, color =colour)
            
            # plot legend
            ax.legend(loc = 'upper right')


    # add title and labels
    fig.suptitle('Average cross-decoding accuracies for different "distances" between training and test session', fontsize = 25)
    ax.set_title("Note: the longer the distance the fewer pairs there are (e.g., 2 pairs for dist 10, 4 pairs for dist 9 ... 20 pairs for dist 1) \n that is, there is more uncertainty in the average accuracy for longer distances", fontsize = 20)

    fig.supylabel('Accuracy (%)')
    fig.supxlabel('Samples')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

def main_plot_generator():
    accuracies_cross = {} # not including testing and training on the same session
    accuracies = {} # including testing and training on the same session

    for parc in ['aparc','aparc.DKTatlas', 'aparc.a2009s', 'sens', 'HCPMMP1']:
        # read in results
        accuracies[parc] = np.load(os.path.join('accuracies', f'cross_decoding_10_LDA_{parc}.npy'), allow_pickle=True)

        # plot all pairs of sessions in one figure
        plot_cross_decoding_matrix(accuracies[parc], parc)

        # plot diagonals per session
        cross_diags_per_sesh(accuracies[parc], save_path = os.path.join('plots', f'cross_decoding_{parc}_diagonals.png'))
        # average over all sessions
        cross_diags_average_sesh(accuracies[parc], save_path = os.path.join('plots', f'cross_decoding_{parc}_diagonals_average.png'))

        # set within session accuracies to nan
        acc1 = accuracies[parc].copy()
        acc1[np.arange(acc1.shape[0]), np.arange(acc1.shape[1]), :, :] = np.nan

        accuracies_cross[parc] = acc1

        # plot average over all conditions and all cross-session pairs
        plt = plot.plot_tgm_fig(np.nanmean(acc1, axis=(0, 1)), vmin=42.5, vmax=57.5, chance_level=chance_level(588*11, alpha = alpha, p = 0.5))
        plt.savefig(os.path.join('plots', f'cross_decoding_{parc}_average.png'))
        
        # plot averaged according to conditions and using cross-session pairs
        plot_train_test_condition(acc1, parc, diff_colour='red')


    # Diagonals of the parcellations together
    plot_diagonals(accuracies_cross, title = 'Diagonals across sessions', save_path = os.path.join('plots', f'diagonals_across.png'))
    plot_diagonals(accuracies, title = 'Diagonals within session', save_path = os.path.join('plots', f'diagonals_within.png'), )



if __name__ == '__main__':
    main_plot_generator()