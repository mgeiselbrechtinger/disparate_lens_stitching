#! /usr/bin/python3

import csv
import subprocess as sp
from pathlib import Path
import matplotlib.pyplot as plt

ALGOS=['sift', 'surf', 'orb', 'akaze', 'brisk', 'harris-laplace', 'asift']
RES_DIR='./results/kp_quad_size/'
HEADER="transform,repeatability,correspondences,source_keypoints,destination_keypoints\n"

def main():
    #plt.figure("Keypoint evaluation")
    fig, axs = plt.subplots(1, 3)
    axs[0].set_ylabel("average repeatability")
    axs[0].set_xlabel("transformation [degrees]")
    axs[1].set_ylabel("average #correspondences")
    axs[1].set_xlabel("transformation [degrees]")
    axs[2].set_ylabel("average #destination-keypoints")
    axs[2].set_xlabel("transformation [degrees]")
    # loop over algorithms
    for algo in ALGOS:
        in_file = open(f"{RES_DIR}bev_kp_eval_{algo}.csv", 'r', newline='') 
        reader = csv.reader(in_file, delimiter=',')
        data_x = list()
        data_y_0 = list()
        data_y_1 = list()
        data_y_2 = list()
        # loop over transforms
        next(reader)
        for row in reader:
            data_x.append(float(row[0]))
            data_y_0.append(float(row[1]))
            data_y_1.append(float(row[2]))
            data_y_2.append(float(row[4]))
            
        data_x = [10*x for x in data_x]
        axs[0].plot(data_x, data_y_0, label=algo)
        axs[1].plot(data_x, data_y_1, label=algo)
        axs[2].plot(data_x, data_y_2, label=algo)
    
    [ax.grid() for ax in axs]
    axs[1].legend()
    plt.show()

if __name__ == "__main__":
    main()

