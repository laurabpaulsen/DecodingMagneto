import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).parents[2]))
from utils.analysis import plot
from utils.analysis.tools import chance_level

from cross_decoding.plot_results import cross_diags_average_sesh, plot_cross_decoding_matrix, cross_diags_per_sesh, cross_diags_average_sesh
import numpy as np



def main():
    alpha = 0.05
    path = Path(__file__).parent
    acc = np.load(path / "accuracies" / "LDA_auto_10.npy")

    # plot all pairs of sessions in one figure
    plot_cross_decoding_matrix(acc, save_path = path / "plots" / "entire_matrix.png")

    # plot diagonals per session
    cross_diags_per_sesh(acc, save_path = path / "plots" / "sens_diagonals.png")
    
    # average over all sessions
    cross_diags_average_sesh(acc, save_path = path / "plots" / "sens_average_diagonals.png", title="Average decoding accuracy given distances between bins")

    # set within session accuracies to nan
    acc1 = acc.copy()
    acc1[np.arange(acc1.shape[0]), np.arange(acc1.shape[1]), :, :] = np.nan


    # plot average over all conditions and all cross-session pairs
    plt = plot.plot_tgm_fig(np.nanmean(acc1, axis=(0, 1)), vmin=40, vmax=60, chance_level=chance_level(588*11, alpha = alpha, p = 0.5))
    plt.savefig(path / "plots" / "sens_average.png")

if __name__ in "__main__":
    main()