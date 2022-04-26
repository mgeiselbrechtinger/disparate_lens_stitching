#! /usr/bin/python3

import csv
import subprocess as sp
from pathlib import Path

ALGOS=['sift', 'surf', 'orb', 'akaze', 'brisk', 'harris-laplace', 'asift']
CMD='./keypoints/build/main'
DATA_DIR='../data/robotcar/'
OUT_DIR='./results/'
HEADER="transform,repeatability,correspondences,source_keypoints,destination_keypoints\n"

def main():
    # loop over algorithms
    for algo in ALGOS:
        out_file = open(f"{OUT_DIR}bev_kp_eval_{algo}.csv", 'w', newline='') 
        out_file.write(HEADER)
        cmd_args = [CMD, "-d", algo]
        # loop over transforms
        tf_path = sorted(Path(f"{DATA_DIR}bev/").glob('*'))
        for i, tf_file in enumerate(tf_path):
            cmd_args_ = cmd_args + [tf_file]
            vals = [0]*4
            # loop over images
            img_path = Path(f"{DATA_DIR}left/").glob('*')
            for j, img_file in enumerate(img_path):
                cmd_args_ = cmd_args + [img_file, tf_file]
                # Execute cmd
                cmd_res = sp.run(cmd_args_, stdout=sp.PIPE, text=True)
                vals = [v + float(s) for (v, s) in zip(vals, cmd_res.stdout.split(','))]

            # average over images
            vals = [str(i)] + [str(round(v/j, 4)) for v in vals]
            line = ','.join(vals)
            out_file.write(f"{line}\n")
            

if __name__ == "__main__":
    main()

