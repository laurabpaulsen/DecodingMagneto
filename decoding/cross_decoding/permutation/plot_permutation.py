import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def plot_values(array, save_path=None, alpha=[0.05]):
    fig, ax = plt.subplots(figsize=(8, 8))

    # set values above the significance level to nan
    array[array > alpha[0]] = np.nan
    im = ax.imshow(array, cmap='Reds_r', origin='lower', vmin=0, vmax=0.06, interpolation='bilinear')

    ax.set_xlabel('Testing time')
    ax.set_ylabel('Training time')
    fig.suptitle('Permutation test', fontsize=20)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('P-value', rotation=-90, va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

def main():
    conditions = [
        {"train1": "vis", "test1": "vis", "train2": "mem", "test2": "mem"},
        {"train1": "mem", "test1": "mem", "train2": "mem", "test2": "vis"},
        {"train1": "vis", "test1": "vis", "train2": "mem", "test2": "vis"},
        {"train1": "vis", "test1": "vis", "train2": "mem", "test2": "vis"},
        {"train1": "vis", "test1": "mem", "train2": "mem", "test2": "mem"},
        {"train1": "vis", "test1": "mem", "train2": "mem", "test2": "vis"},
        {"train1": "vis", "test1": "mem", "train2": "vis", "test2": "vis"}]
    
    # loop over conditions
    for cond in conditions:
        try:
            # load data
            p_values = np.load(f"sens_p_values_{cond['train1']}{cond['test1']}_{cond['train2']}{cond['test2']}.npy")

            # plot
            plot_values(p_values, save_path = f"sens_p_values_{cond['train1']}{cond['test1']}_{cond['train2']}{cond['test2']}.png")
        
        # just skip if the file does not exist
        except FileNotFoundError:
            print(f"File not found")
            print(f"{cond['train1']}{cond['test1']}_{cond['train2']}{cond['test2']}.npy")
            continue

if __name__ in "__main__":
    main()