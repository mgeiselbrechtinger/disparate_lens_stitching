#! /usr/bin/python3

import json
import sys
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

sys.path.append("../src/")
sys.path.append("./src/")

from stitch_extracted import stitcher

ALGOS=['sift', 'orb', 'akaze', 'r2d2', 'keynet', 'sosnet', 'hardnet'] # TODO surf, brisk

AUC_THRESHOLD = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=False, help="Path to output directory")
    parser.add_argument('--in_dir', required=True, help="Path to input directory")
    args = parser.parse_args()


    #plot_keypoints(args.in_dir)
    plot_accuracy(args.in_dir)

            
def plot_accuracy(in_dir):
    fig, axs =  plt.subplots(1, 1, figsize=(10, 4))
    # loop over algorithms
    for algo in ALGOS:

        ratios = range(2, 7)
        if algo in ['keynet', 'r2d2']: 
            ratios = range(2, 6)

        with open(f"{in_dir}/res_{algo}.json", 'r') as in_file:
            data = json.load(in_file)
        
            mAA = data['mAA']
            
            axs.plot(ratios, mAA, label=algo)

    plt.legend()
    axs.set_xlabel('disparity level')
    axs.set_ylabel('mAA')
    plt.show()

def plot_keypoints(in_dir):
    fig, axs =  plt.subplots(1, 2, figsize=(10, 4))
    # loop over algorithms
    for algo in ALGOS:

        ratios = range(2, 7)
        if algo in ['keynet', 'r2d2']: 
            ratios = range(2, 6)

        with open(f"{in_dir}/res_{algo}.json", 'r') as in_file:
            data = json.load(in_file)
        
            n_kp = data['kpts']
            kp_ratio = data['kp_ratio']
            
            axs[0].plot(ratios, n_kp, label=algo)
            axs[1].plot(ratios, kp_ratio, label=algo)

    plt.legend()
    axs[0].set_xlabel('disparity level')
    axs[1].set_xlabel('disparity level')
    axs[0].set_ylabel('# keypoints')
    axs[1].set_ylabel('relevant kp ratio')
    plt.show()

if __name__ == "__main__":
    main()

