################################################################################
#
#   Stitch two images
#
#
################################################################################

import argparse

import cv2
import numpy as np
import time
import math
import sys

sys.path.append("../../src/")
sys.path.append("./src/")

from compose_mosaic import compose
import feature_matchers as fm

func_dict = { 'sift'    : fm.sift_detect_and_match, 
              'orb'     : fm.orb_detect_and_match,
              'surf'    : fm.surf_detect_and_match,
              'mix'     : fm.mix_detect_and_match,
              'akaze'   : fm.akaze_detect_and_match,
              'brisk'   : fm.brisk_detect_and_match,
              'hardnet' : fm.hardnet_detect_and_match }

stats = { 'keypoints'   : 0,
          'matches'     : 0,
          'ratio'       : 1,
          'error'       : 0 }

RANSAC_THRESHOLD = 4.0
MATCH_THRESHOLD = 4.0

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_names', nargs=2, help="Paths to src-images")
    parser.add_argument('hg_name', help="Paths to ground truth")
    parser.add_argument('-d', '--detector', choices=func_dict.keys(), required=True, help="Determine which feature detector to use")
    parser.add_argument('-r', '--ratio', default=3.0, type=float, help="Ratio (>3) between src and dest image")
    parser.add_argument('-t', '--top', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-o', '--outfile', help="Path for stitched image")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-b', '--benchmark', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.benchmark:
        args.verbose = False

    # Load files
    img_src = cv2.imread(args.img_names[0], cv2.IMREAD_COLOR)
    img_dest = cv2.imread(args.img_names[1], cv2.IMREAD_COLOR)
    if img_src is None or img_dest is None:
        raise OSError(-1, "Could not open file.", args.img_names)

    H_ref = np.loadtxt(args.hg_name, delimiter=',')

    # Downscale to fit synthetic experiment
    #dsize = img_src.shape[1]//3, img_src.shape[0]//3
    #img_src = cv2.resize(img_src, dsize, 0)
    #img_dest = cv2.resize(img_dest, dsize, 0)
    #H_ref[0, 2] /= 3
    #H_ref[1, 2] /= 3
    #H_ref[2, 0] *= 3
    #H_ref[2, 1] *= 3

    r = 3/args.ratio
    if r > 1:
        raise ValueError("Ratio has to be larger equal to 3")

    stats['ratio'] = args.ratio

    # Downscale source image to mimic higher resolution of destination image
    h, w, _ = img_src.shape
    dsize = round(w*r), round(h*r)
    img_src = cv2.resize(img_src, dsize, interpolation=cv2.INTER_AREA)
    S = np.float32([[1/r, 0, 0], [0, 1/r, 0], [0, 0, 1]])

    # Crop destination image to mimic restricted FOV
    h, w, _ = img_dest.shape
    img_dest = img_dest[int(h - h*r)//2 : int(h + h*r)//2, int(w - w*r)//2 : int(w + w*r)//2]
    A = np.float32([[1, 0, (w - w*r)/2], [0, 1, (h - h*r)/2], [0, 0, 1]])
    
    # Rescale groundtruth homography
    H_gt = np.linalg.inv(A).dot(H_ref.dot(S))
    H_gt /= H_gt[2, 2]

    img_src_g = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_dest_g = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)

    # Detect and Match
    mstart = time.time()
    matches, kp_src, kp_dest = func_dict[args.detector](img_src_g, img_dest_g)
    mduration = time.time() - mstart

    # Check keypoint stats
    n_kp_dest = len(kp_dest)
    # Check if source kpts are in destination FOV
    pts_src = cv2.KeyPoint_convert(kp_src).astype(int)
    roi_mask = cv2.warpPerspective(255*np.ones_like(img_dest_g), np.linalg.inv(H_gt), dsize)
    kp_mask = (roi_mask[pts_src[:, 1], pts_src[:, 0]] == 255)
    n_kp_src = np.count_nonzero(kp_mask)
    stats['keypoints'] = min(n_kp_src, n_kp_dest)

    if args.verbose:
        print(f"Found {n_kp_src} relevant keypoints in source image and {n_kp_dest} in destination image")

        kp_src_img = cv2.drawKeypoints(img_src, np.array(kp_src)[np.logical_not(kp_mask)], None, [0, 0, 255])
        kp_src_img = cv2.drawKeypoints(img_src, np.array(kp_src)[kp_mask], 
                                       kp_src_img, [0, 255, 0], cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
        kp_dest_img = cv2.drawKeypoints(img_dest, kp_dest, None, [0, 255, 0])
        kp_img = np.concatenate((kp_src_img, kp_dest_img), axis=1)
        cv2.namedWindow("keypoints", cv2.WINDOW_NORMAL)        
        cv2.setWindowProperty("keypoints", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("keypoints", kp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # Extract data from DMatch objects
    matches_dist = np.zeros(len(matches))
    matches_qidx = np.zeros(len(matches), dtype=np.int32)
    matches_tidx = np.zeros(len(matches), dtype=np.int32)
    for i, m in enumerate(matches):
        matches_dist[i] = m.distance
        matches_qidx[i] = m.queryIdx
        matches_tidx[i] = m.trainIdx

    # Find inliers and calculate error using ground truth
    pts_src = cv2.KeyPoint_convert(kp_src, matches_qidx)
    pts_dest = cv2.KeyPoint_convert(kp_dest, matches_tidx)
    pts_src_ref = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
    dists = np.linalg.norm(pts_dest - pts_src_ref, axis=1) 
    inlier_mask = (dists < MATCH_THRESHOLD)
    stats['matches'] = np.count_nonzero(inlier_mask)
    
    if args.verbose:
        print(f"Matched {len(matches)} from {len(kp_src)} source keypoints to {len(kp_dest)} destination keypoints in {mduration:03f}s")
        print(f"Match inlier ratio: {np.count_nonzero(inlier_mask)/inlier_mask.size : 2f}")

        draw_params = dict(flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                           matchColor=[0, 0, 255],
                           matchesMask=[(0 if x else 1) for x in inlier_mask])
        matches_img = cv2.drawMatches(img_src, kp_src, img_dest, kp_dest,
                                      matches, None, **draw_params)
        draw_params = dict(flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG,
                           matchColor=[0, 255, 0],
                           matchesMask=[(1 if x else 0) for x in inlier_mask])
        cv2.drawMatches(img_src, kp_src, img_dest, kp_dest,
                        matches, matches_img, **draw_params)

        cv2.namedWindow("matches", cv2.WINDOW_NORMAL)        
        cv2.setWindowProperty("matches", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('matches', matches_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    # Estimate Homography
    hstart = time.time()

    ransac_params = dict(method=cv2.USAC_PROSAC,
                         ransacReprojThreshold=RANSAC_THRESHOLD,
                         maxIters=10000,
                         confidence=0.999999)

    # Sort keypoints for PROSAC
    if ransac_params['method'] == cv2.USAC_PROSAC:
        sort_args = matches_dist.argsort()
        pts_src = pts_src[sort_args]
        pts_dest = pts_dest[sort_args]
        pts_src_ref = pts_src_ref[sort_args]

    H, inlier_mask = cv2.findHomography(pts_src, pts_dest, **ransac_params)
    hduration = time.time() - hstart

    if H is None:
        raise ValueError("Homography estimation failed")

    inlier_mask = inlier_mask.ravel()
    pts_src_est = cv2.perspectiveTransform(pts_src[inlier_mask].reshape(-1, 1, 2), H).reshape(-1, 2)
    dists = np.linalg.norm(pts_src_ref[inlier_mask] - pts_src_est, axis=1)
    stats['error'] = np.mean(dists)

    if args.benchmark:
        print(f"Stats: {stats}")
        return


    if args.verbose:
        print(f"RANSAC inlier ratio: {np.count_nonzero(inlier_mask)/inlier_mask.size}")
        print(f"Estimated homography with {args.detector.upper()} in {hduration:03f}s")
        print(H)

        res = compose(img_dest, [img_src], [H], base_on_top=args.top)

        cv2.namedWindow("composition", cv2.WINDOW_NORMAL)        
        cv2.imshow("composition", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Done in {mduration+hduration:03f}s")
        print(f"Scale change h, w: {res.shape[0]/img_src.shape[0], res.shape[1]/img_src.shape[1]}")
        print(stats)


if __name__ == '__main__':
    main()

