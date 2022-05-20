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

from feature_matchers import match
from compose_mosaic import compose

def stitcher(img_src, img_des, kp_src, kp_dest, des_src, des_dest, H_gt,
             des_type, match_threshold, ransac_params, top=False, verbose=False): 
    # Performance statistics
    stats = { 'src_keypoints'   : -1,
              'dest_keypoints'  : -1,
              'roi_keypoints'   : -1,
              'matches'         : -1,
              'correct_matches' : -1,
              'match_threshold' : 0,
              'inliers'         : -1,
              'correct_inliers' : -1,
              'mean_error'      : float('inf'),
              'max_error'       : float('inf'),
              'iou'             : 0,
              'width'           : 0,
              'height'          : 0 } 

    # Matching
    mstart = time.time()
    matches = match(kp_src, des_src, kp_dest, des_dest, des_type=des_type)
    mduration = time.time() - mstart

    h, w, _ = img_dest.shape

    ratio = np.sqrt(H_gt[0,0]*H_gt[1,1])
    dsize = int(ratio*w), int(ratio*h)
    r = 1/ratio
    
    # Keypoint stats
    stats['src_keypoints'] = len(kp_src)
    stats['dest_keypoints'] = len(kp_dest)
    # Check if source kpts are in destination FOV
    pts_src = cv2.KeyPoint.convert(kp_src).astype(int)
    roi_mask = cv2.warpPerspective(255*np.ones_like(img_dest[...,0]), np.linalg.inv(H_gt), dsize)
    kp_mask = (roi_mask[pts_src[:, 1], pts_src[:, 0]] == 255)
    n_kp_src = np.count_nonzero(kp_mask)
    stats['roi_keypoints'] = n_kp_src

    if verbose:
        print(f"Found {n_kp_src} relevant keypoints in source image and {len(kp_dest)} in destination image")

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

    # Matching stats
    stats['matches'] = len(matches)
    # Find inliers using ground truth
    pts_src = cv2.KeyPoint.convert(kp_src, matches_qidx)
    pts_dest = cv2.KeyPoint.convert(kp_dest, matches_tidx)
    pts_proj_gt = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
    dists = np.linalg.norm(pts_proj_gt - pts_dest, axis=1) 
    match_mask = (dists <= match_threshold)
    stats['match_threshold'] = match_threshold
    stats['correct_matches'] = np.count_nonzero(match_mask)

    if verbose:
        print(f"Matched {len(matches)} from {len(kp_src)} source keypoints to {len(kp_dest)} destination keypoints")
        print(f"Match inlier ratio: {np.count_nonzero(match_mask)/match_mask.size : 2f}")

        draw_params = dict(flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                           matchColor=[0, 0, 255],
                           matchesMask=[(0 if x else 1) for x in match_mask])
        matches_img = cv2.drawMatches(img_src, kp_src, img_dest, kp_dest,
                                      matches, None, **draw_params)
        draw_params = dict(flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG,
                           matchColor=[0, 255, 0],
                           matchesMask=[(1 if x else 0) for x in match_mask])
        cv2.drawMatches(img_src, kp_src, img_dest, kp_dest,
                        matches, matches_img, **draw_params)

        cv2.namedWindow("matches", cv2.WINDOW_NORMAL)        
        cv2.setWindowProperty("matches", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('matches', matches_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # Early abort
    if stats['correct_matches'] <= 4:
        return stats

    # Estimate Homography
    hstart = time.time()

    # Sort keypoints for PROSAC
    if ransac_params['method'] == cv2.USAC_PROSAC:
        sort_args = matches_dist.argsort()
        match_mask = match_mask[sort_args]
        pts_src = pts_src[sort_args]
        pts_dest = pts_dest[sort_args]
        pts_proj_gt = pts_proj_gt[sort_args]

    H, inlier_mask = cv2.findHomography(pts_src, pts_dest, **ransac_params)
    hduration = time.time() - hstart

    if H is None:
        raise ValueError("Homography estimation failed")

    inlier_mask = inlier_mask.ravel()
    stats['inliers'] = np.count_nonzero(inlier_mask)
    stats['correct_inliers'] = np.count_nonzero(np.bitwise_and(inlier_mask, match_mask))
    pts_proj_est = cv2.perspectiveTransform(pts_src[inlier_mask].reshape(-1, 1, 2), H).reshape(-1, 2)
    dists = np.linalg.norm(pts_proj_gt[inlier_mask] - pts_proj_est, axis=1)
    stats['mean_error'] = np.mean(dists)/r
    stats['max_error'] = np.amax(dists)/r
    # Calculate IoU of projection
    roi = 255*np.ones_like(img_src[...,0])
    roi_proj_gt = cv2.warpPerspective(roi, H_gt, (w, h))
    roi_proj_est = cv2.warpPerspective(roi, H, (w, h))
    iou = np.count_nonzero(np.bitwise_and(roi_proj_gt, roi_proj_est))/np.count_nonzero(np.bitwise_or(roi_proj_gt, roi_proj_est))
    stats['iou'] = iou

    if verbose:
        print(f"RANSAC inlier ratio: {np.count_nonzero(inlier_mask)/inlier_mask.shape[0]}")
        print(f"Estimated homography in {hduration:03f}s")
        print(H)

        res = compose(img_dest, [img_src], [H], base_on_top=top)

        cv2.namedWindow("composition", cv2.WINDOW_NORMAL)        
        cv2.imshow("composition", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Done in {mduration+hduration:03f}s")
        print(f"Scale change h, w: {res.shape[0]/img_src.shape[0], res.shape[1]/img_src.shape[1]}")

    return stats

if __name__ == '__main__':
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', help="Path to image")
    parser.add_argument('matcher_dir', help="Path to image")
    parser.add_argument('-t', '--top', type=bool, default=False)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = parser.parse_args()

    # Load files
    img_src = cv2.imread(args.img_dir + '/c_short.png', cv2.IMREAD_COLOR)
    img_dest = cv2.imread(args.img_dir + '/c_long.png', cv2.IMREAD_COLOR)
    if img_src is None or img_dest is None:
        raise OSError(-1, "Could not open files.", args.img_name)

    H_gt = np.loadtxt(args.img_dir + '/h_short2long.csv', delimiter=',')
    
    # Load Keypoints and Descriptors
    src_pts =   np.load(args.matcher_dir + '/c_short_kpt.npy')
    dest_pts =   np.load(args.matcher_dir + '/c_long_kpt.npy')
    src_kp =    cv2.KeyPoint.convert(src_pts[:, :2])
    dest_kp =   cv2.KeyPoint.convert(dest_pts[:, :2])
    src_des =   np.load(args.matcher_dir + '/c_short_dsc.npy')
    dest_des =   np.load(args.matcher_dir + '/c_long_dsc.npy')
    
    # Setup parameters
    match_threshold = 4.0
    ransac_params = dict(method=cv2.USAC_PROSAC,
                         ransacReprojThreshold=4.0,
                         maxIters=10000,
                         confidence=0.999999)

    dest_type = cv2.NORM_L2

    stats = stitcher(img_src, img_dest, 
                     src_kp, dest_kp, 
                     src_des, dest_des, 
                     H_gt, dest_type, 
                     match_threshold, 
                     ransac_params,
                     top=args.top, verbose=args.verbose)

    print(stats)


