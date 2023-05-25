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

alpha = 0.05

# set parameters for all plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.dpi'] = 300


def determine_linestyle(train_condition, test_condition):
    if train_condition == test_condition:
        return "-"
    elif train_condition != test_condition:
        return "--"
    
def determine_colour(i, j):
    colors = sns.color_palette("hls", 12)

    return colors[abs(i-j)]

def add_diagonal_line(ax, colour = 'red'):
    """
    Adds a diagonal line to a plot
    """
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = colour, linestyle = '--', alpha = 0.5, linewidth = 1)


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

def x_axis_seconds(ax):
    """
    Changes the x axis to seconds
    """
    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

def y_axis_percent(ax):
    """
    Changes the y axis seconds
    """
    ax.set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

def plot_cross_decoding_matrix(acc, save_path = None):
    fig, axs = plt.subplots(acc.shape[0], acc.shape[1], figsize = (30, 30))
    
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            axs[i, j] = plot.plot_tgm_ax(acc[i, j], ax=axs[i, j], vmin=35, vmax=65)
            axs[i, j].set_title(f'train:{i+1}, test:{j+1}')
    
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path)
    plt.close()



def plot_sig_clusters(ax, array_name):
    # add significant clusters to plot as contour
    cluster_array = np.load(f'permutation/sens_sig_clusters_{array_name}.npy', allow_pickle=True)

    # plot contour where value is 1
    ax.contour(cluster_array, levels = [0.5], colors = "k", linewidths = 0.3, alpha = 0.7)

    # somehow ax.contour messes with the axis, so we need to reset it
    ax.set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

    return ax


def add_dif_label(ax, label, colour = 'red'):
    ax.text(-0.08, 1.1, label, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right', color = colour, alpha = 0.7)

def plot_train_test_condition(acc, parc, vmin = 40, vmax = 60, diff_colour = 'darkblue'):
    fig, axs = plt.subplots(4, 3, figsize = (12, 12*4/3), dpi = 300)
    
    vis = np.array([0, 1, 2, 3, 4, 5, 6])
    mem = np.array([7, 8, 9, 10])
    
    n_trials_vis = 588 * len(vis)
    n_trials_mem = 588 * len(mem)

    vmin_diff = -5
    vmax_diff = 5
    
    # VIS VIS
    vis_vis = np.nanmean(acc[vis,:, :, :][:, vis, :, :], axis = (0, 1))

    axs[1, 0] = plot.plot_tgm_ax(vis_vis, ax=axs[1, 0], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_vis, alpha = alpha, p = 0.5), title ='train: vis,  test:vis'.upper())
    axs[1,0].text(-0.1, 1.1, 'A', transform=axs[1,0].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

    # MEM MEM
    mem_mem = np.nanmean(acc[mem,:, :, :][:, mem, :, :], axis = (0, 1))
    axs[1, 1] = plot.plot_tgm_ax(mem_mem, ax=axs[1, 1], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_mem, alpha = alpha, p = 0.5), title="train:mem, test:mem".upper())
    axs[1,1].text(-0.1, 1.1, 'B', transform=axs[1,1].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

    # VIS MEM
    vis_mem = acc[vis,:, :, :][:, mem, :, :].mean(axis = (0, 1))
    axs[2, 0] = plot.plot_tgm_ax(vis_mem, ax=axs[2, 0], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_mem,alpha = alpha, p = 0.5), title = 'train:vis, test:mem'.upper())
    axs[2,0].text(-0.1, 1.1, 'C', transform=axs[2,0].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

    # MEM VIS
    mem_vis = acc[mem, :, :, :][:, vis, :, :].mean(axis = (0, 1))
    axs[2, 1] = plot.plot_tgm_ax(mem_vis, ax=axs[2, 1], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_vis, alpha = alpha, p = 0.5), title='train:mem, test:vis'.upper())
    axs[2,1].text(-0.1, 1.1, 'D', transform=axs[2,1].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

    ### DIFFERENCE PLOTS ###
    colour_map_diff = "PuOr_r" # sns.diverging_palette(220, 20, s = 70, l = 70, as_cmap=True)
    # difference between test and train condition (vis - mem)
    axs[1, 2] = plot.plot_tgm_ax(np.array(vis_vis - mem_mem), ax=axs[1, 2], vmin=vmin_diff, vmax=vmax_diff, cmap = colour_map_diff)
    plot_sig_clusters(axs[1, 2], "visvis_memmem")
    add_dif_label(axs[1, 2], 'A-B', colour = diff_colour)

    # difference between test and train condition
    axs[2, 2] = plot.plot_tgm_ax(vis_mem - mem_vis, ax=axs[2, 2], vmin=vmin_diff, vmax=vmax_diff, cmap = colour_map_diff)
    axs[2, 2]= plot_sig_clusters(axs[2, 2], "vismem_memvis")
    add_dif_label(axs[2, 2], 'C-D', colour = diff_colour)

    # difference between vis_vis and vis_mem
    axs[3, 0] = plot.plot_tgm_ax(vis_vis - vis_mem, ax=axs[3, 0], vmin=vmin_diff, vmax=vmax_diff, cmap = colour_map_diff)
    plot_sig_clusters(axs[3, 0], "visvis_vismem")
    add_dif_label(axs[3, 0], 'A-C', colour = diff_colour)

    # difference between mem_mem and mem_vis
    axs[3, 1] = plot.plot_tgm_ax(mem_mem - mem_vis, ax=axs[3, 1], vmin=vmin_diff, vmax=vmax_diff, cmap = colour_map_diff)
    axs[3, 1] = plot_sig_clusters(axs[3, 1], "memmem_memvis")
    add_dif_label(axs[3, 1], 'B-D', colour = diff_colour)

    # difference between vis_vis and mem_vis
    axs[3, 2] = plot.plot_tgm_ax(vis_vis - mem_vis, ax=axs[3, 2], vmin=vmin_diff, vmax=vmax_diff, cmap = colour_map_diff)
    axs[3, 2] = plot_sig_clusters(axs[3, 2], "visvis_memvis")
    add_dif_label(axs[3, 2], 'A-D', colour = diff_colour)

    # difference between vis_mem and mem_mem
    axs[0, 2] = plot.plot_tgm_ax(vis_mem - mem_mem, ax=axs[0, 2], vmin=vmin_diff, vmax=vmax_diff, cmap = colour_map_diff)
    axs[0, 2] =plot_sig_clusters(axs[0, 2], "vismem_memmem")
    add_dif_label(axs[0, 2], 'C-B', colour = diff_colour)


    for ax in axs[[2, 3, 3, 3, 1, 3, 0], [2, 2, 0, 1, 2, 2, 2]].flatten(): # difference plots
        x_axis_seconds(ax)
        change_spine_colour(ax, diff_colour)
        add_diagonal_line(ax, colour = diff_colour)

    # plot colourbars in the first two columns of the first row
    colour_loc = [axs[1, 0].images[0], axs[1, 2].images[0]]
    labels = ['ACCURACY (%)', 'DIFFERENCE (%)']
    for i, ax in enumerate(axs[0, :2]):
        # remove the axis
        ax.axis('off')
        # add colourbar
        cb = plt.colorbar(colour_loc[i], ax = ax, orientation = 'horizontal', pad = 0.9, shrink = 0.8, location = "bottom")
        cb.ax.set_title(labels[i], fontsize = 14)

        if i == 1:
            change_spine_colour(cb.ax, diff_colour)

    axs[2,0].set_ylabel('TRAIN TIME (s)', fontsize=14)
    axs[-1,1].set_xlabel('TEST TIME (s)', fontsize=14)

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

    ax.set_xlabel('time (ms)'.upper())
    ax.set_ylabel('accuracy'.upper())
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path)


def cross_diags_average_sesh(accuracies, SE=False, save_path=None, title=None):
    diagonals = np.diagonal(accuracies, axis1=2, axis2=3)

    # multiply by 100 to get percentages
    diagonals = diagonals * 100


    order = [0, 1, 2, 3, 7, 8, 9, 10, 4, 5, 6]

    vis_inds = [0, 1, 2, 3, 7, 8, 9, 10]
    mem_inds = [4, 5, 6, 7]

    fig, ax = plt.subplots(3, 1, figsize = (12, 14), dpi = 300)

    for ax_ind, cond in enumerate([vis_inds, mem_inds, None]):
        # create 11 by 11 matrix with distances between sessions
        # empty matrix
        distances = np.zeros((11, 11))

        # fill in the matrix
        for i in range(11):
            for j in range(11):
                distances[i, j] = abs(order[i]-order[j])
            
        if cond:
            distances = distances[cond, :][:, cond]
        
        # loop over all distances
        for dist in range(11):
            # skip distance 0
            if dist == 0:
                pass
            else:
                # get the indices of the sessions that are dist away from each other
                indices = np.argwhere(distances == dist)

                if indices.shape[0] == 0:
                    continue

                # get the average accuracy for each pair of sessions that are dist away from each other
                tmp = np.array([diagonals[i, j] for i, j in indices])
                n_pairs = tmp.shape[0]

                # standard deviation

                tmp_acc = tmp.mean(axis = 0)

                colour = determine_colour(12, dist)
                
                # plot tmp acc
                ax[ax_ind].plot(tmp_acc, label = f'{dist} ({n_pairs} pairs)', linewidth = 1, color = colour)
                
                if SE: 
                    # standard deviation
                    tmp_std = tmp.std(axis = 0)
                    
                    # standard error
                    tmp_std = tmp_std / np.sqrt(tmp.shape[0])

                    # plot standard error
                    ax[ax_ind].fill_between(np.arange(0, 250), (tmp_acc - tmp_std), (tmp_acc + tmp_std), alpha = 0.1, color =colour)
                
            # plot legend
            ax[ax_ind].legend(loc = 'upper right', title = "Distance".upper())
            ax[ax_ind].set_ylabel(['Visual', 'Memory', 'Combined'][ax_ind].upper())
                
            # set x axis to seconds
            x_axis_seconds(ax[ax_ind])


    # add title and labels
    if title:
        fig.suptitle(title.upper())

    fig.supylabel('Average cross-decoding accuracy (%)'.upper())
    fig.supxlabel('Time (s)'.upper())
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

def main_plot_generator():
    accuracies_cross = {} # not including testing and training on the same session
    accuracies = {} # including testing and training on the same session

    for parc in ["sens"]:#, "HCPMMP1"]: #['aparc','aparc.DKTatlas', 'aparc.a2009s', 'sens', 'HCPMMP1']:
        
        # read in results
        accuracies[parc] = np.load(os.path.join('accuracies', f'cross_decoding_10_LDA_{parc}.npy'), allow_pickle=True)

        # plot all pairs of sessions in one figure
        #plot_cross_decoding_matrix(accuracies[parc], save_path = os.path.join('plots', f'cross_decoding_{parc}_matrix.png'))

        # average over all sessions
        cross_diags_average_sesh(accuracies[parc], save_path = os.path.join('plots', f'cross_decoding_{parc}_diagonals_average.png'))

        # set within session accuracies to nan
        acc1 = accuracies[parc].copy()
        acc1[np.arange(acc1.shape[0]), np.arange(acc1.shape[1]), :, :] = np.nan

        accuracies_cross[parc] = acc1

        # plot average over all conditions and all cross-session pairs
        plt = plot.plot_tgm_fig(np.nanmean(acc1, axis=(0, 1)), vmin=40, vmax=60, chance_level=chance_level(588*11, alpha = alpha, p = 0.5), )
        plt.savefig(os.path.join('plots', f'cross_decoding_{parc}_average.png'))

        # plot averaged according to conditions and using cross-session pairs
        plot_train_test_condition(acc1, parc, diff_colour='darkblue')


    # Diagonals of the parcellations together
    #plot_diagonals(accuracies_cross, title = 'Diagonals across sessions', save_path = os.path.join('plots', f'diagonals_across.png'))
    #plot_diagonals(accuracies, title = 'Diagonals within session', save_path = os.path.join('plots', f'diagonals_within.png'), )



if __name__ == '__main__':
    main_plot_generator()