#! /usr/bin/python3

import json
import argparse
import numpy as np
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt


KP_ALGOS = ['sift', 'surf', 'orb', 'brisk', 'akaze', 'r2d2', 'keynet']
DSC_ALGOS = KP_ALGOS + ['sosnet', 'hardnet']

# Global style settings
mpl.style.use('ggplot')
COLOR_MAP = plt.get_cmap('gist_rainbow')
COLORS = 0.8*COLOR_MAP(np.linspace(0, 1, len(DSC_ALGOS)))
color_cycler = (cycler(color=COLORS))
plt.rc('axes', prop_cycle=color_cycler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=False, help="Path to output directory")
    parser.add_argument('--in_dir', required=True, help="Path to input directory")
    args = parser.parse_args()

    # load data
    data = dict()
    for algo in DSC_ALGOS:
        with open(f"{args.in_dir}/res_{algo}.json", 'r') as in_file:
            data[algo] = json.load(in_file)

    plot_keypoints(data)
    plot_matching_score(data)
    plot_accuracy(data)

            
def plot_accuracy(data):
    x = np.arange(2, 7)
    fig, axs =  plt.subplots(1, 1)
    # loop over algorithms
    for algo in DSC_ALGOS:
        mAA = data[algo]['mAA']
        axs.plot(x, mAA, label=algo)

    axs.legend()
    axs.set_xlabel('disparity level')
    axs.set_ylabel('mAA')
    plt.show()


def plot_keypoints(data):
    fig, axs =  plt.subplots(1, 2, tight_layout=True)

    # Indicate FOV ratio for reference
    x = np.arange(2, 7)
    axs[1].plot(x, 1/x**2, color='darkgray', linestyle='dashed', linewidth=0.7)

    for algo in KP_ALGOS:
        n_kp = data[algo]['kpts']
        kp_ratio = data[algo]['kp_ratio']
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

    for algo in DSC_ALGOS:
        ms = data[algo]['matching_score']
        ir = data[algo]['inlier_ratio']
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

