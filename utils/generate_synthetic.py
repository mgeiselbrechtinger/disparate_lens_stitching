################################################################################
#
#   Synthesize two images with different focal lengths
#
#
################################################################################

import argparse

import cv2
import numpy as np

DATA_PATH="data/tracks/"
OUTPUT_PATH="data/synthetic/"

if __name__ == '__main__':
    rng = np.random.default_rng(seed=79)
    # Handle arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('img_name', help="Path to source-image")
    #parser.add_argument('out_dir', help="Path to output directory")
    #parser.add_argument('-r', '--ratio', default=2.0, type=float, help="Ratio between src and dest image")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()


    for seq in range(10):

        img_name = f"{DATA_PATH}/seq{seq}/c40_0.png"

        # Load file
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(-1, "Could not open file.", img_name)

        h, w, _ = img.shape

        for ratio in range(2, 7):
            r = 1/ratio
            dsize = round(w*r), round(h*r)

            # Downscale source image to mimic higher resolution of destination image
            img_src = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
            S = np.float32([[1/r, 0, 0], [0, 1/r, 0], [0, 0, 1]])

            # Crop destination image to mimic restricted FOV
            img_dest = img[round(h - h*r)//2 : round(h + h*r)//2, round(w - w*r)//2 : round(w + w*r)//2]
            A = np.float32([[1, 0, (w - w*r)/2], [0, 1, (h - h*r)/2], [0, 0, 1]])

            # Add noise to destination image
            noise = rng.normal(0, 0.1, img_dest.shape)
            contrast = rng.uniform(0.70, 1.30)
            img_dest = contrast*(img_dest/255) + noise
            img_dest = np.clip(255*img_dest, 0, 255).astype(np.uint8)

            # Groundtruth homography
            H_gt = np.linalg.inv(A).dot(S)
            H_gt /= H_gt[2, 2]

            if args.verbose: 
                viz = np.concatenate((img_src, img_dest), axis=1)
                cv2.namedWindow("composition", cv2.WINDOW_NORMAL)        
                cv2.imshow("composition", viz)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            cv2.imwrite(f"{OUTPUT_PATH}/seq{seq}/ratio{ratio}/c_short.png", img_src)
            cv2.imwrite(f"{OUTPUT_PATH}/seq{seq}/ratio{ratio}/c_long.png", img_dest)
            np.savetxt(f"{OUTPUT_PATH}/seq{seq}/ratio{ratio}/h_short2long.csv", H_gt, delimiter=',')

