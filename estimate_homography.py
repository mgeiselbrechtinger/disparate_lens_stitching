################################################################################
#
#   Estimate homography between image pair based on different feature matchers
#
#
################################################################################

import argparse

import cv2
import numpy as np

import feature_matchers as fm

func_dict = { 'sift'    : fm.sift_detect_and_match, 
              'orb'     : fm.orb_detect_and_match,
              'akaze'   : fm.akaze_detect_and_match,
              'brisk'   : fm.brisk_detect_and_match,
              'hardnet' : fm.hardnet_detect_and_match }

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_names', nargs=2, help="Paths to src- and dest-image)")
    parser.add_argument('-o', '--outfile', dest='hg_name', help="Path for homography csv-file")
    parser.add_argument('-d', '--detector', choices=func_dict.keys(), help="Determine which feature detector to use")
    parser.add_argument('-e', '--evaluate', dest='gt_name', help="Path to ground truth homography csv-file")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Load files
    img_src = cv2.imread(args.img_names[0], cv2.IMREAD_COLOR)
    img_dest = cv2.imread(args.img_names[1], cv2.IMREAD_COLOR)
    if img_src is None or img_dest is None:
        raise OSError(-1, "Could not open file.", args.img_names[0], args.img_names[1])

    # Scale
    #img_src = cv2.pyrDown(img_src)
    #img_src = cv2.pyrUp(img_src)
    #img_dest = cv2.pyrDown(img_dest)
    #img_dest = cv2.pyrUp(img_dest)
    # Crop
    #h, w, c = img_src.shape
    #img_src = img_src[h//2-200:h//2+200, w//2-400:w//2+400]
    #img_dest = img_dest[h//2-200:h//2+200, w//2-400:w//2+400]
    #img_src[:h//2] = 0

    img_src_g = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_dest_g = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)


    # Detect and Match
    matches, kp_src, kp_dest = func_dict[args.detector](img_src_g, img_dest_g)

    if args.verbose:
        print(f"Matched {len(matches)} from {len(kp_src)} source keypoints to {len(kp_dest)} destination keypoints")

        draw_params = dict(flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS |
                                 cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        matches_img = cv2.drawMatches(img_src, kp_src, img_dest, kp_dest,
                                      matches[::5], None, **draw_params)

        cv2.namedWindow("matches", cv2.WINDOW_NORMAL)        
        cv2.setWindowProperty("matches", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('matches', matches_img)
        cv2.waitKey(0)


    pts_src = list()
    pts_dest = list()
    #matches = sorted(matches, key=lambda m: m.distance) # Sorting required for PROSAC 
    for m in matches:
        pts_src.append(kp_src[m.queryIdx].pt)
        pts_dest.append(kp_dest[m.trainIdx].pt)

    H, inlier_mask = cv2.findHomography(np.array(pts_src, dtype=np.float32), 
                                        np.array(pts_dest, dtype=np.float32), 
                                        cv2.RANSAC, 7.0)

    if args.verbose or args.gt_name != None:
        print(f"Estimated homography with {args.detector.upper()}")
        print(H)

    if args.hg_name != None:
        np.savetxt(args.hg_name, H, delimiter=',')

    # Compare estimate with ground truth 
    if args.gt_name != None:
        H_ref = np.loadtxt(args.gt_name, delimiter=',')
        print("Ground truth homography")
        print(H_ref)
        print("Elementwise error")
        print(np.abs(H_ref/H - 1))
        print(f"Max difference: {np.amax(np.abs(H_ref - H)):05f}")


if __name__ == '__main__':
    main()

