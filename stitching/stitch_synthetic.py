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

from compose_mosaic import compose
import feature_matchers as fm

func_dict = { 'sift'    : fm.sift_detect_and_match, 
              'orb'     : fm.orb_detect_and_match,
              'akaze'   : fm.akaze_detect_and_match,
              'brisk'   : fm.brisk_detect_and_match,
              'hardnet' : fm.hardnet_detect_and_match }

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name', help="Paths to src-image)")
    parser.add_argument('hg_names', nargs=2, help="Paths to src- and dest-transformation)")
    parser.add_argument('-d', '--detector', choices=func_dict.keys(), required=True, help="Determine which feature detector to use")
    parser.add_argument('-t', '--top', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-o', '--outfile', help="Path for stitched image")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Load files
    img = cv2.imread(args.img_name, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(-1, "Could not open file.", args.img_name)

    # Transform images
    h, w, _ = img.shape
    H_src = np.loadtxt(args.hg_names[0], delimiter=',')
    src_scale = 2
    H_src[0, 2] /= src_scale 
    H_src[1, 2] /= src_scale 
    H_src[2, 0] *= src_scale 
    H_src[2, 1] *= src_scale 
    img_src = cv2.pyrDown(img)
    img_src = cv2.warpPerspective(img_src, H_src, (w//src_scale, h//src_scale))

    H_dest = np.loadtxt(args.hg_names[1], delimiter=',')
    img_dest = img
    img_dest = cv2.warpPerspective(img_dest, H_dest, (w, h))
    img_dest = (img_dest*0.75).astype(np.uint8)
    img_dest = img_dest[h//4:3*h//4, w//4:3*w//4]
    img_dest_uc = img_dest
    img_dest = cv2.copyMakeBorder(img_dest, h//4, h//4, w//4, w//4, cv2.BORDER_CONSTANT, 0)

    H_src_inv = np.linalg.inv(H_src)
    H_dest_inv = np.linalg.inv(H_dest)

    img_src_g = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_dest_g = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)

    # Detect and Match
    mstart = time.time()
    matches, kp_src, kp_dest = func_dict[args.detector](img_src_g, img_dest_g)
    mduration = time.time() - mstart

    # Extract data from DMatch objects
    matches_dist = np.zeros(len(matches))
    matches_qidx = np.zeros(len(matches), dtype=np.int32)
    matches_tidx = np.zeros(len(matches), dtype=np.int32)
    for i, m in enumerate(matches):
        matches_dist[i] = m.distance
        matches_qidx[i] = m.queryIdx
        matches_tidx[i] = m.trainIdx

    if args.verbose:
        print(f"Matched {len(matches)} from {len(kp_src)} source keypoints to {len(kp_dest)} destination keypoints in {mduration:03f}s")
        
        # Find inliers using ground truth
        inlier_threshold = 5
        pts_src = cv2.KeyPoint_convert(kp_src, matches_qidx)
        pts_dest = cv2.KeyPoint_convert(kp_dest, matches_tidx)
        pts_src_ref = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H_src_inv)*src_scale
        pts_dest_ref = cv2.perspectiveTransform(pts_dest.reshape(-1, 1, 2), H_dest_inv)
        inlier_mask = (np.linalg.norm(pts_src_ref - pts_dest_ref, axis=2) < inlier_threshold)
        print(f"Match inlier ratio: {np.count_nonzero(inlier_mask)/inlier_mask.size : 2f}")

        draw_params = dict(flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                           matchColor=[0, 255, 0],
                           matchesMask=[(1 if x else 0) for x in inlier_mask])
        matches_img = cv2.drawMatches(img_src, kp_src, img_dest, kp_dest,
                                      matches, None, **draw_params)
        draw_params = dict(flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG,
                           matchColor=[0, 0, 255],
                           matchesMask=[(0 if x else 1) for x in inlier_mask])
        cv2.drawMatches(img_src, kp_src, img_dest, kp_dest,
                        matches, matches_img, **draw_params)

        cv2.namedWindow("matches", cv2.WINDOW_NORMAL)        
        cv2.setWindowProperty("matches", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('matches', matches_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Estimate Homography
    hstart = time.time()

    ransac = cv2.USAC_PROSAC
    # Sort keypoints for PROSAC
    if ransac == cv2.USAC_PROSAC:
        sort_args = matches_dist.argsort()
        pts_src = pts_src[sort_args]
        pts_dest = pts_dest[sort_args]

    H, inlier_mask = cv2.findHomography(pts_src, pts_dest, ransac, 7.0)
    hduration = time.time() - hstart

    if H is None:
        raise ValueError("Homography estimation failed")

    if args.verbose:
        print(f"Estimated homography with {args.detector.upper()} in {hduration:03f}s")
        print(H)

    img_src = img_src
    img_dest = img_dest_uc

    A = np.float32([[1, 0, -w//4], [0, 1, -h//4], [0, 0, 1]])
    res = compose(img_dest, [img_src], [A.dot(H)], base_on_top=args.top)

    cv2.namedWindow("composition", cv2.WINDOW_NORMAL)        
    cv2.imshow("composition", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Done in {mduration+hduration:03f}s")

    if args.verbose:
        print(f"Scale change h, w: {res.shape[0]/img_src.shape[0], res.shape[1]/img_src.shape[1]}")

if __name__ == '__main__':
    main()

