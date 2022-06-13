#! /usr/bin/python3

import json
import argparse
import numpy as np
from cycler import cycler
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


KP_ALGOS = ['sift', 'orb', 'surf', 'brisk', 'akaze', 'r2d2', 'keynet']
DSC_ALGOS = KP_ALGOS + ['sosnet', 'hardnet']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=False, help="Path to output directory")
    parser.add_argument('--in_dir', required=True, help="Path to input directory")
    args = parser.parse_args()

    # Load data
    data = dict()
    in_paths = sorted(Path(args.in_dir).glob('*'))
    for p in in_paths:
        with p.open(mode='r') as f:
            algo = p.name.split('.')[0]
            data[algo] = json.load(f)

    # Set styles
    mpl.style.use('ggplot') # Style sheet
    cmap = plt.get_cmap('gist_rainbow') # Line colors
    # Prevent line color repetition
    colors = 0.8*cmap(np.linspace(0, 1, len(data.keys())))
    color_cycler = (cycler(color=colors)) 
    plt.rc('axes', prop_cycle=color_cycler)

    # Plot data
    #plot_keypoints(data)
    #plot_matching_score(data)
    plot_accuracy(data)
    #plot_accuracy_comp(data)


def plot_accuracy_comp(data):
    # Set dashed lines and same color for algorithms optimized version
    cmap = plt.get_cmap('gist_rainbow') # Line colors
    colors = 0.8*cmap(np.linspace(0, 1, len(data.keys())//2))
    color_cycler = (cycler(linestyle=['-', '--'])*cycler(color=colors))
    plt.rc('axes', prop_cycle=color_cycler)
    plot_accuracy(dict(sorted(data.items(), key=lambda k: len(k[0].replace('+', '-').split('-')))))

            
def plot_accuracy(data):
    x = np.arange(2, 7)
    fig, axs =  plt.subplots(1, 1)
    # loop over algorithms
    for algo, d in data.items():
        mAA = d['mAA']
        axs.plot(x, mAA, label=algo)

    axs.legend()
    axs.set_xlabel('disparity level')
    axs.set_ylabel('mAA')
    plt.show()


def plot_keypoints(data):
    fig, axs =  plt.subplots(1, 2, tight_layout=True)

    # Indicate FOV ratio for reference
    x = np.arange(2, 7)
    axs[1].plot(x, 1/x**2, color='black', linestyle='dashed', linewidth=0.7)

    for algo, d in data.items():
        if all([algo.find(kp_algo)  == -1 for kp_algo in KP_ALGOS]):
            axs[0].plot(x[0], 0)
            axs[1].plot(x[0], 0)

        else:
            n_kp = d['kpts']
            kp_ratio = d['kp_ratio']
            axs[0].plot(x, n_kp, label=algo)
            axs[1].plot(x, kp_ratio, label=algo)


    plt.legend()
    axs[0].set_xlabel('disparity level')
    axs[0].set_ylabel('# keypoints')
    axs[1].set_xlabel('disparity level')
    axs[1].set_ylabel('relevant kp ratio')
    plt.show()


def plot_matching_score(data):
    fig, axs =  plt.subplots(1, 2, tight_layout=True)
    x = np.arange(2, 7)

    for algo, d in data.items():
        ms = d['matching_score']
        ir = d['inlier_ratio']
        axs[0].plot(x, ms, label=algo)
        axs[1].plot(x, ir, label=algo)


    plt.legend()
    axs[0].set_xlabel('disparity level')
    axs[0].set_ylabel('matching score')
    axs[1].set_xlabel('disparity level')
    axs[1].set_ylabel('inlier ratio')
    plt.show()


if __name__ == "__main__":
    main()

