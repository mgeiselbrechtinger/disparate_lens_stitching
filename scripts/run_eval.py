#! /usr/bin/python3

import json
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

sys.path.append("../src/")
sys.path.append("./src/")

from stitch_synthetic import stitcher

ALGOS=['brisk', 'orb', 'sift', 'surf', 'akaze']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help="Path to output directory")
    parser.add_argument('in_dir', help="Path to input directory")
    args = parser.parse_args()

    match_threshold = 4.0
    ransac_params = dict(method=cv2.USAC_PROSAC,
                         ransacReprojThreshold=4.0,
                         maxIters=10000,
                         confidence=0.999999)


    # loop over algorithms
    for algo in ALGOS:
        ratios = np.arange(2, 5.5, 0.5)
        rank = np.zeros_like(ratios)
        error = np.zeros_like(ratios)
        # loop over image sequences
        seq_paths = Path(args.in_dir).glob('*')
        for i, sp in enumerate(seq_paths):
            in_file = f"{sp}/c40_0.png"
            img = cv2.imread(in_file, 1)
            # loop over ratios
            for j, r in enumerate(ratios):
                try:
                    s = stitcher(img, algo, r, match_threshold, ransac_params)

                except e:
                    print(e)

                else:
                    if s['iou'] >= 0.6:
                        rank[j] += 1
                        error[j] += s['mean_error']

        rank /= (i+1)
        error /= (i+1)
        print(rank)
        print(error)
        with open(f"{args.out_dir}/res_{algo}.json", 'w') as of:
            json.dump(rank.tolist(), of)
    
        print(f"Finished evaluation {algo}")
            

if __name__ == "__main__":
    main()

