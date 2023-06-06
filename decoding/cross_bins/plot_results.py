import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
from utils.analysis import plot
from utils.analysis.tools import chance_level

from cross_decoding.plot_results import determine_colour, x_axis_seconds
import numpy as np
import matplotlib.pyplot as plt



def cross_diags_average_sesh(acc_vis, acc_mem, acc_all, SE=False, save_path=None, title=None):
    
    diagonals_vis = np.diagonal(acc_vis, axis1=2, axis2=3) * 100
    diagonals_mem = np.diagonal(acc_mem, axis1=2, axis2=3) * 100
    diagonals_all = np.diagonal(acc_all, axis1=2, axis2=3) * 100


    fig, ax = plt.subplots(3, 1, figsize = (12, 14), dpi = 300)

    for ax_ind, acc in enumerate([diagonals_vis, diagonals_mem, diagonals_all]):
        # create 11 by 11 matrix with distances between sessions
        # empty matrix
        distances = np.zeros((acc.shape[0], acc.shape[0]))
        order = np.arange(acc.shape[0])

        # fill in the matrix
        for i in range(11):
            for j in range(11):
                distances[i, j] = abs(order[i]-order[j])
                    
        # loop over all distances
        for dist in range(acc.shape[0]):
            # skip distance 0
            if dist == 0:
                pass
            else:
                # get the indices of the sessions that are dist away from each other
                indices = np.argwhere(distances == dist)

                if indices.shape[0] == 0:
                    continue

                # get the average accuracy for each pair of sessions that are dist away from each other
                tmp = np.array([acc[i, j] for i, j in indices])
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


def plot_tgms(vis_acc, mem_acc, all_acc, save_path=None, title=None):
    fig, ax = plt.subplots(1, 3, figsize = (14, 6), dpi = 300)

    for ax_ind, (acc, subtitle) in enumerate(zip([vis_acc, mem_acc, all_acc], ["VISUAL", "MEMORY", "COMBINED"])):

        # set within session accuracies to nan
        acc1 = acc.copy()
        acc1[np.arange(acc1.shape[0]), np.arange(acc1.shape[1]), :, :] = np.nan

        # plot
        plot.plot_tgm_ax(np.nanmean(acc1, axis =(0, 1)), ax = ax[ax_ind], vmin=40, vmax=60)

        # set title
        ax[ax_ind].set_title(subtitle)


    if title:
        fig.suptitle(title.upper())
    
    if save_path:
        plt.savefig(save_path)


def main():
    alpha = 0.05
    path = Path(__file__).parent
    acc_all = np.load(path / "accuracies" / "LDA_auto_10_all_11.npy")
    acc_vis = np.load(path / "accuracies" / "LDA_auto_10_visual_11.npy")
    acc_mem = np.load(path / "accuracies" / "LDA_auto_10_memory_11.npy")
    
    # average over all sessions
    cross_diags_average_sesh(acc_vis, acc_mem, acc_all, save_path = path / "plots" / "sens_average_diagonals_bins.png", title="Average decoding accuracy given distances between bins")


    # plot average
    plot_tgms(acc_vis, acc_mem, acc_all, save_path = path / "plots" / "sens_average_tgm_bins.png")

    

if __name__ in "__main__":
    main()