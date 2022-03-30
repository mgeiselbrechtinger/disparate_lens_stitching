################################################################################
#
#   Applies homographies to images and composes them around base image
#
#
################################################################################

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

REF = "homography" # "homography" | "sift" | "orb" 

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('base_img_name', help="Paths to base image")
    parser.add_argument('img_names', nargs='+', help="Paths to stitch images")
    parser.add_argument('hg_names', nargs='+', help="Paths to homography csv-files")
    args = parser.parse_args()

    if len(args.img_names) != len(args.hg_names):
        print("Same number of image and homography files required")
        sys.exit(-1)

    # Load bas image
    base_img = cv2.imread(args.base_img_name, cv2.IMREAD_UNCHANGED)
    if base_img is None:
        print("Couldn't load image")
        sys.exit(-1)

    # Create common frame from base image
    frame_offset_x = base_img.shape[1]//2
    frame_offset_y = base_img.shape[0]//4 
    frame = cv2.copyMakeBorder(base_img, frame_offset_y, frame_offset_y, 
                                         frame_offset_x, frame_offset_x, 
                                         borderType=cv2.BORDER_CONSTANT, value=0)

    # Align remaining images to frame
    for img_name, hg_name in zip(args.img_names, args.hg_names):
        # Load image and Homography
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Couldn't load image")
            sys.exit(-1)

        H = np.loadtxt(hg_name, delimiter=',')
        if H.shape != (3, 3):
            print("Homography of wrong format")
            sys.exit(-1)

        # Add alpha channel with all ones and enlarge image to frame size
        alpha = 255*np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
        img = cv2.copyMakeBorder(img, frame_offset_y, frame_offset_y, 
                                      frame_offset_x, frame_offset_x, 
                                      borderType=cv2.BORDER_CONSTANT, value=0)
    
        # Apply homography and overlay on frame
        img = cv2.warpPerspective(img, H, (frame.shape[1], frame.shape[0])) 
        mask = (img[:, :, 3] == 255)
        frame[mask, :] = img[mask, :3]


    cv2.imshow("composition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
