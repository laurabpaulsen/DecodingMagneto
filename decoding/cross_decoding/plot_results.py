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
from scipy.signal import find_peaks
from pathlib import Path

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
    ax.set_xticks(np.arange(0, 251, step=50))
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])

def y_axis_percent(ax):
    """
    Changes the y axis seconds
    """
    ax.set_yticks(np.arange(0, 101, step=20))
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])
def plot_cross_decoding_matrix(acc, save_path = None):
    fig, axs = plt.subplots(acc.shape[0], acc.shape[1], figsize = (12, 12))
    
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


    for ax in axs[:, -1]:
        # remove x ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # set title of all axes on first row
    for i, ax in enumerate(axs[0, :]):
        ax.set_title(f'Session {i+1}')

    # set title of all axes on first column
    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(f'Session {i+1}')


    # axis of in the last column
    #for ax in axs[:, -1]:
    #    ax.axis('off')

    # increase space between plots
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path)
    plt.close()



def add_dif_label(ax, label, colour = 'red'):
    ax.text(-0.02, 1.05, label, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right', color = colour, alpha = 0.7)

def add_tgm_label(ax, label, colour = 'black'):
    ax.text(-0.05, 1.05, label, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right', color = colour, alpha = 0.7)


def cross_diags_average_sesh(accuracies, SE=False, save_path=None, title=None):
    diagonals = np.diagonal(accuracies, axis1=2, axis2=3)

    # multiply by 100 to get percentages
    diagonals = diagonals * 100


    order = [0, 1, 2, 3, 7, 8, 9, 10, 4, 5, 6]

    vis_inds = [0, 1, 2, 3, 7, 8, 9, 10]
    mem_inds = [4, 5, 6, 7]

    fig, ax = plt.subplots(3, 1, figsize = (12, 14), dpi = 300)

    for ax_ind, cond in enumerate([vis_inds]):
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


def average_diagonal(acc, alpha = None, acc_pair = None, save_path = None):
    """

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi = 300)

    diag = np.diagonal(acc, axis1=0, axis2=1)

    # multiply by 100 to get percentages
    diag = diag * 100

    if acc_pair is not None:
        for i in range(acc_pair.shape[0]):
            for j in range(acc_pair.shape[1]):
                if i == j:
                    continue
                else:
                    ax.plot(np.diagonal(acc_pair[i, j], axis1=0, axis2=1)*100, color = 'lightblue', alpha = 0.1)

    # detect peaks 
    peaks, _ = find_peaks(diag, height = 54, distance = 10)

    ax.plot(diag)
    ax.plot(peaks, diag[peaks], "x", color = 'k')

    time_peaks = peaks/250
    value_peaks = diag[peaks]

    if alpha:
        # detect where significance is reached
        sig = np.where(diag > chance_level(588*11, alpha = alpha, p = 0.5)*100)[0]
        sig_time = sig[0]/250



        print(f"Time peaks: {time_peaks}")
        print(f"Value peaks: {value_peaks}")
        print(f"Time sig: {sig_time}")
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


def average_diagonal_within(acc, alpha = None, acc_pair = None, save_path = None):
    """

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi = 300)

    diag = np.diagonal(acc, axis1=0, axis2=1)

    # multiply by 100 to get percentages
    diag = diag * 100
    if acc_pair is not None:
        for i in range(acc_pair.shape[2]):
            ax.plot(np.diagonal(acc_pair[:, :, i], axis1=0, axis2=1)*100, color = 'lightblue', alpha = 0.2)

    # detect peaks 
    peaks, _ = find_peaks(diag, height = 54, distance = 10)

    ax.plot(diag)
    ax.plot(peaks, diag[peaks], "x", color = 'k')

    time_peaks = peaks/250
    value_peaks = diag[peaks]


    # plot a dashed line with chance level
    if alpha:
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


def corr_distance_decoding(acc, save_path = None):
    """
    Plot the correlation between distance between sessions and decoding accuracy per timepoint
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi = 300)

    session_days = [0, 1, 7, 8, 145, 159, 161]

    diagonals = np.diagonal(acc, axis1=-2, axis2=-1)
    print(diagonals.shape)

    correlations = np.zeros(250)

    for t in range(diagonals.shape[-1]):
        # get data from the given timepoint
        data_timepoint = diagonals[..., t]
        
        distances = []
        accuracies = []
        for i in range(7):
            for j in range(7):
                if i == j:
                    continue
                else:
                    # get the distance between the sessions
                    distance = abs(session_days[i] - session_days[j])

                    # get the accuracy
                    accuracy = data_timepoint[i, j]

                    distances.append(distance)
                    accuracies.append(accuracy)
        
        # calculate the correlation
        correlations[t] = np.corrcoef(distances, accuracies)[0, 1]

    ax.plot(correlations)

    if save_path:
        plt.savefig(save_path)

def main_plot_generator():
    path = Path(__file__)
    output_path = path.parents[0] / 'plots' 
    
    # create output path if it does not exist
    if not output_path.exists():
        output_path.mkdir()

    accuracies_cross = {} # not including testing and training on the same session
    accuracies = {} # including testing and training on the same session
    accuracies_within = {} # only testing and training on the same session

    
    # read in results
    accuracies = np.load(os.path.join('accuracies', f'cross_decoding_10_LDA_sens.npy'), allow_pickle=True)

    # average over the last dimension (cv folds)
    accuracies = np.mean(accuracies, axis = -1)


    accuracies_within = np.diagonal(accuracies, axis1=0, axis2=1)



    # plot all pairs of sessions in one figure
    plot_cross_decoding_matrix(accuracies, save_path = output_path / 'cross_decoding_matrix.png')

    # plot all pairs in the of the first 4 sessions in one figure
    plot_cross_decoding_matrix(accuracies[:3, :3, :, :], save_path = os.path.join('plots', f'cross_decoding_matrix_first4.png'))

    # average over all sessions
    #cross_diags_average_sesh(accuracies, save_path = os.path.join('plots', f'cross_decoding_diagonals_average.png'))
        
    # set within session accuracies to nan
    acc1 = accuracies.copy()
    acc1[np.arange(acc1.shape[0]), np.arange(acc1.shape[1]), :, :] = np.nan

    accuracies_cross = acc1

    # plot average over all conditions and all cross-session pairs

    # average over cross-session pairs
    avg = np.nanmean(acc1, axis=(0, 1))

    plt = plot.plot_tgm_fig(avg, vmin=35, vmax=65, chance_level=chance_level(588*11, alpha = alpha, p = 0.5), cbar_loc='right')
    plt.savefig(os.path.join('plots', f'cross_decoding_average.png'))

    # plot the average over all sessions
    average_diagonal(
        avg,
        alpha,
        acc_pair=accuracies_cross,
        save_path = os.path.join('plots', f'diagonals_across.png'))
        
    # plot the within session diagonals

    # average within session pairs
    avg_within = np.nanmean(accuracies_within, axis = -1)
        
    average_diagonal_within(
        avg_within,
        alpha,
        acc_pair=accuracies_within,
        save_path = os.path.join('plots', f'diagonals_within.png'))

    plt = plot.plot_tgm_fig(avg_within, vmin=35, vmax=65, chance_level=chance_level(588*11, alpha = alpha, p = 0.5), cbar_loc='right')
    plt.savefig(os.path.join('plots', f'within_decoding_average.png'))

    # plot the difference between within and cross session pairs
    difference = avg - avg_within
    plt = plot.plot_tgm_fig(difference, vmin=-4, vmax=4, cmap = "PuOr_r", cbar_loc='right')

    plt.savefig(os.path.join('plots', f'difference_decoding_average.png'))

    # diagonal of the difference
    average_diagonal(
        difference,
        save_path = os.path.join('plots', f'diagonals_difference.png'))


if __name__ == '__main__':
    main_plot_generator()