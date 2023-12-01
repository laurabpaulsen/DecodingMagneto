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
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.dpi'] = 300



def determine_linestyle(train_condition, test_condition):
    if train_condition == test_condition:
        return "-"
    elif train_condition != test_condition:
        return "--"
    
def determine_colour(i, j):
    colors = sns.color_palette("hls", 12)

    return colors[abs(i-j)]

def add_diagonal_line(ax, colour = 'red', linewidth = 1):
    """
    Adds a diagonal line to a plot
    """
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color = colour, linestyle = '--', alpha = 0.5, linewidth = linewidth)


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
    fig, axs = plt.subplots(acc.shape[0], acc.shape[1], figsize = (12, 12), gridspec_kw={'width_ratios': [1, 1, 1]})
    
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            #if i == j:
            #    axs[i, j].axis('off')
            #    plot.plot_tgm_ax(np.zeros((250, 250)), ax=axs[i, j], vmin=-1, vmax=0.5, cmap='Greys')

            plot.plot_tgm_ax(acc[i, j], ax=axs[i, j], vmin=35, vmax=65)
            # add diagonal line
            add_diagonal_line(axs[i, j], colour='grey', linewidth=2)


            # remove x ticks
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    
    # add colour bar
    #colour_loc = axs[1, 0].images[0]
    #cb = plt.colorbar(colour_loc, ax = axs[1, -1], location = 'right')
    #cb.ax.set_ylabel('Accuracy (%)')

    for ax in axs[:, -1]:
        # remove x ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # set title of all axes on first row
    for i, ax in enumerate(axs[0, :]):
        ax.set_title(f'Session {i+1}')

    # set title of all axes on first column
    for i, ax in enumerate(axs[:, 0]):
        print(f"Session {i+1}")
        ax.set_ylabel(f'Session {i+1}')


    # axis of in the last column
    #for ax in axs[:, -1]:
    #    ax.axis('off')

    # increase space between plots
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
    ax.text(-0.02, 1.05, label, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right', color = colour, alpha = 0.7)

def add_tgm_label(ax, label, colour = 'black'):
    ax.text(-0.05, 1.05, label, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right', color = colour, alpha = 0.7)

def plot_train_test_condition_new(acc, parc, vmin = 40, vmax = 60, diff_colour = 'darkblue'):
    fig, axs = plt.subplots(3, 3, figsize = (14, 12), dpi = 300, gridspec_kw={'height_ratios': [1, 1, 0.20]})
    
    vis = np.array([0, 1, 2, 3, 4, 5, 6])
    mem = np.array([7, 8, 9, 10])
    
    n_trials_vis = 588 * len(vis)
    n_trials_mem = 588 * len(mem)

    vmin_diff = -5
    vmax_diff = 5
    
    # VIS VIS
    vis_vis = np.nanmean(acc[vis,:, :, :][:, vis, :, :], axis = (0, 1))

    plot.plot_tgm_ax(vis_vis, ax=axs[0, 0], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_vis, alpha = alpha, p = 0.5), title ='visual -> visual')
    add_diagonal_line(axs[0, 0], colour = "grey")

    # MEM VIS
    mem_vis = acc[mem, :, :, :][:, vis, :, :].mean(axis = (0, 1))
    plot.plot_tgm_ax(mem_vis, ax=axs[0, 1], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_vis, alpha = alpha, p = 0.5), title='memory -> visual')
    add_diagonal_line(axs[0, 1], colour = "grey")

    # MEM MEM
    mem_mem = np.nanmean(acc[mem,:, :, :][:, mem, :, :], axis = (0, 1))
    plot.plot_tgm_ax(mem_mem, ax=axs[1, 0], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_mem, alpha = alpha, p = 0.5), title="memory -> memory")
    add_diagonal_line(axs[1, 0], colour = "grey")

    # VIS MEM
    vis_mem = acc[vis,:, :, :][:, mem, :, :].mean(axis = (0, 1))
    plot.plot_tgm_ax(vis_mem, ax=axs[1, 1], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_mem,alpha = alpha, p = 0.5), title = 'visual -> memory')
    add_diagonal_line(axs[1, 1], colour = "grey")
    

    ### DIFFERENCE PLOTS ###
    colour_map_diff = "PuOr_r" # sns.diverging_palette(220, 20, s = 70, l = 70, as_cmap=True)

    # difference between vis_vis and mem_vis
    plot.plot_tgm_ax(vis_vis - mem_vis, ax=axs[0, 2], vmin=vmin_diff, vmax=vmax_diff, cmap = colour_map_diff, title = '(visual -> visual) \n - (memory -> visual)')
    plot_sig_clusters(axs[0, 2], "visvis_memvis")

    # difference between vis_mem and mem_mem
    plot.plot_tgm_ax(mem_mem - vis_mem, ax=axs[1, 2], vmin=vmin_diff, vmax=vmax_diff, cmap = colour_map_diff, title = '(memory -> memory) \n - (visual -> memory)')
    plot_sig_clusters(axs[1, 2], "vismem_memmem")


    for ax in [axs[1,2], axs[0, 2]]:
        x_axis_seconds(ax)
        change_spine_colour(ax, diff_colour)
        add_diagonal_line(ax, colour = diff_colour)

    #plot colourbars in the first two columns of the first row
    colour_loc = [axs[1, 0].images[0], axs[1, 2].images[0]]
    labels = ['Accuracy (%)', 'Difference (%)']
    for i, ax in enumerate(axs[-1, 1:3]):
        # remove the axis
        ax.axis('off')
        # add colourbar
        cb = plt.colorbar(colour_loc[i], ax = ax, orientation = 'horizontal', pad = 0.9, shrink = 0.8)
        cb.ax.set_title(labels[i])

        if i == 1:
            change_spine_colour(cb.ax, diff_colour)

    # for all the bottom plots
    #for ax in axs[-1, :]:
    #    ax.set_xlabel('Test time (s)')
    #for ax in axs[:, 0]:
    #    ax.set_ylabel('TRAIN TIME (s)', fontsize=14)

    for ax in axs.flatten():
        ax.set_xlabel('Test time (s)')
        ax.set_ylabel('Train time (s)')

    # remove last axis
    axs[-1, 0].axis('off')

    # add space between plots
    plt.tight_layout(h_pad = 3)

    plt.savefig(os.path.join('plots', f'cross_decoding_{parc}_average_vis_mem_new.png'))
    plt.close()




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
    add_diagonal_line(axs[1, 0], colour = "grey")

    # MEM MEM
    mem_mem = np.nanmean(acc[mem,:, :, :][:, mem, :, :], axis = (0, 1))
    axs[1, 1] = plot.plot_tgm_ax(mem_mem, ax=axs[1, 1], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_mem, alpha = alpha, p = 0.5), title="train:mem, test:mem".upper())
    axs[1,1].text(-0.1, 1.1, 'B', transform=axs[1,1].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    add_diagonal_line(axs[1, 1], colour = "grey")

    # VIS MEM
    vis_mem = acc[vis,:, :, :][:, mem, :, :].mean(axis = (0, 1))
    axs[2, 0] = plot.plot_tgm_ax(vis_mem, ax=axs[2, 0], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_mem,alpha = alpha, p = 0.5), title = 'train:vis, test:mem'.upper())
    axs[2,0].text(-0.1, 1.1, 'C', transform=axs[2,0].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    add_diagonal_line(axs[2, 0], colour = "grey")
    

    # MEM VIS
    mem_vis = acc[mem, :, :, :][:, vis, :, :].mean(axis = (0, 1))
    axs[2, 1] = plot.plot_tgm_ax(mem_vis, ax=axs[2, 1], vmin=vmin, vmax=vmax, chance_level=chance_level(n_trials_vis, alpha = alpha, p = 0.5), title='train:mem, test:vis'.upper())
    axs[2,1].text(-0.1, 1.1, 'D', transform=axs[2,1].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    add_diagonal_line(axs[2, 1], colour = "grey")

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


def average_diagonal_permutation(acc, alpha, save_path = None):
    """

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi = 300)

    diag = np.diagonal(acc, axis1=0, axis2=1)

    # multiply by 100 to get percentages
    diag = diag * 100

    ax.plot(diag)

    # plot a dashed line with chance level
    cl = chance_level(588*11, alpha = alpha, p = 0.5)*100

    ax.plot([0, 250], [cl, cl], linestyle = '--', color = 'k', alpha = 0.5)

    ax.set_xlabel('Time (s)'.upper())
    ax.set_ylabel('Accuracy (%)'.upper())

    # set x lim
    ax.set_xlim([0, 250])

    # x axis in seconds
    x_axis_seconds(ax)

    if save_path:
        plt.savefig(save_path)

def main_plot_generator():
    accuracies_cross = {} # not including testing and training on the same session
    accuracies = {} # including testing and training on the same session

    for parc in ["sens"]:#, "HCPMMP1"]: #['aparc','aparc.DKTatlas', 'aparc.a2009s', 'sens', 'HCPMMP1']:
        
        # read in results
        accuracies[parc] = np.load(os.path.join('accuracies', f'cross_decoding_10_LDA_{parc}.npy'), allow_pickle=True)
        print(accuracies[parc].shape)
        # plot all pairs of sessions in one figure
        #plot_cross_decoding_matrix(accuracies[parc], save_path = os.path.join('plots', f'cross_decoding_{parc}_matrix.png'))

        # plot all pairs in the of the first 4 sessions in one figure
        #plot_cross_decoding_matrix(accuracies[parc][:3, :3, :, :], save_path = os.path.join('plots', f'cross_decoding_{parc}_matrix_first4.png'))

        # average over all sessions
        #cross_diags_average_sesh(accuracies[parc], save_path = os.path.join('plots', f'cross_decoding_{parc}_diagonals_average.png'))
        
        # set within session accuracies to nan
        acc1 = accuracies[parc].copy()
        acc1[np.arange(acc1.shape[0]), np.arange(acc1.shape[1]), :, :] = np.nan

        accuracies_cross[parc] = acc1

        # plot average over all conditions and all cross-session pairs

        # average over cross-session pairs
        avg = np.nanmean(acc1, axis=(0, 1))
        plt = plot.plot_tgm_fig(avg, vmin=40, vmax=60, chance_level=chance_level(588*11, alpha = alpha, p = 0.5), cbar_loc='right')
        plt.savefig(os.path.join('plots', f'cross_decoding_{parc}_average.png'))

        # plot the average over all sessions
        average_diagonal_permutation(avg, alpha, save_path = os.path.join('plots', f'diagonals_across_perm.png'))
        

        # plot averaged according to conditions and using cross-session pairs
        #plot_train_test_condition(acc1, parc, diff_colour='darkblue')

        #plot_train_test_condition_new(acc1, parc, diff_colour='darkblue')


    # Diagonals of the parcellations together
    #plot_diagonals(accuracies_cross, title = 'Diagonals across sessions', save_path = os.path.join('plots', f'diagonals_across.png'))
    #plot_diagonals(accuracies, title = 'Diagonals within session', save_path = os.path.join('plots', f'diagonals_within.png'), )



if __name__ == '__main__':
    main_plot_generator()