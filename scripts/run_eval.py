#! /usr/bin/python3

import json
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

sys.path.append("../src/")
sys.path.append("./src/")
feature_dir = "synthetic_tracks"

from stitch_extracted import stitcher

ALGOS = ['sift']
#ALGOS += ['brisk', 'orb', 'akaze', 'sift', 'hardnet', 'sosnet'] 
#ALGOS += ['r2d2', 'keynet'] # Requires previous extraction
#ALGOS += ['surf']           # Requires non-free opencv build

AUC_THRESHOLD = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, help="Path to output directory")
    parser.add_argument('--in_dir', required=True, help="Path to input directory")
    args = parser.parse_args()

    match_threshold = 4.0
    ransac_params = dict(method=cv2.USAC_PROSAC,
                         ransacReprojThreshold=4.0,
                         maxIters=10000,
                         confidence=0.999999)


    sequences = range(10)

    # loop over algorithms
    for algo in ALGOS:

        ratios = range(2, 7)

        ms = np.zeros((len(ratios), len(sequences)))
        kpts = np.zeros((len(ratios), len(sequences)))
        kp_ratios = np.zeros((len(ratios), len(sequences)))
        grid_errs = np.zeros((len(ratios), len(sequences)))
        inlier_ratios = np.zeros((len(ratios), len(sequences)))

        for i, r in enumerate(ratios):

            for s in sequences:
                data_path = f"{args.in_dir}/seq{s}/ratio{r}/"
                img_src = cv2.imread(data_path + '/c_short.png', cv2.IMREAD_COLOR)
                img_dest = cv2.imread(data_path + '/c_long.png', cv2.IMREAD_COLOR)
                if img_src is None or img_dest is None:
                    raise OSError(-1, "Could not open files.", data_path)

                H_gt = np.loadtxt(data_path + '/h_short2long.csv', delimiter=',')
                
                feature_path = f"{algo}/{feature_dir}/seq{s}/ratio{r}/"

                try:
                    stats = stitcher(img_src, img_dest, 
                                     H_gt, algo, feature_path,
                                     r, match_threshold, 
                                     ransac_params)

                except Exception as e:
                    print(e)

                else:
                    kpts[i, s] = min(stats['dest_keypoints'], stats['src_keypoints'])
                    kp_ratios[i, s] = stats['roi_keypoints']/min(stats['dest_keypoints'], stats['src_keypoints'])
                    if stats['roi_keypoints'] > 0:
                        ms[i, s] = stats['correct_matches']/stats['roi_keypoints']
                    err = stats['grid_error']
                    inlier_ratios[i, s] = stats['correct_inliers']/(stats['matches'] - stats['correct_inliers'])
                    
                    err = np.ones((AUC_THRESHOLD-1, 1))*err[None, :] 
                    mask = np.arange(1, AUC_THRESHOLD)[:, None] * np.ones((1, 4)) 
                    grid_errs[i, s] = np.sum(np.mean(err < mask, axis=1))


        data_out = dict()
        data_out['ratios'] = [r for r in ratios]
        data_out['kpts'] = np.mean(kpts, axis=1).tolist()
        data_out['kp_ratio'] = np.mean(kp_ratios, axis=1).tolist()
        data_out['mAA'] = (np.mean(grid_errs, axis=1)/AUC_THRESHOLD).tolist()
        data_out['matching_score'] = np.mean(ms, axis=1).tolist()
        data_out['inlier_ratio'] = np.mean(inlier_ratios, axis=1).tolist()

        with open(f"{args.out_dir}/{algo}.json", 'w') as of:
            json.dump(data_out, of)
    
        print(f"Finished evaluation {algo}")
            

if __name__ == "__main__":
    main()

