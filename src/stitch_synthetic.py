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
              'surf'    : fm.surf_detect_and_match,
              'akaze'   : fm.akaze_detect_and_match,
              'brisk'   : fm.brisk_detect_and_match,
              'hardnet' : fm.hardnet_detect_and_match }

def stitcher(img, detector, ratio, match_threshold, ransac_params, top=False, verbose=False): 
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
              'ratio'           : 1,
              'width'           : 0,
              'height'          : 0 } 

    # Transform images with alpha layers for easy composition
    h, w, _ = img.shape
    r = 1/ratio
    stats['width'] = w
    stats['height'] = h
    stats['ratio'] = r
    dsize = round(w*r), round(h*r)

    # Downscale source image to mimic higher resolution of destination image
    img_src = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
    img_src = np.concatenate((img_src, 255*np.ones_like(img_src[..., :1])), axis=2)
    S = np.float32([[1/r, 0, 0], [0, 1/r, 0], [0, 0, 1]])

    # Crop destination image to mimic restricted FOV
    img_dest = (img*0.75).astype(np.uint8)
    img_dest = img_dest[round(h - h*r)//2 : round(h + h*r)//2, round(w - w*r)//2 : round(w + w*r)//2]
    img_dest = np.concatenate((img_dest, 255*np.ones_like(img_dest[..., :1])), axis=2)
    A = np.float32([[1, 0, (w - w*r)/2], [0, 1, (h - h*r)/2], [0, 0, 1]])
    
    # Groundtruth homography
    H_gt = np.linalg.inv(A).dot(S)
    H_gt /= H_gt[2, 2]

    # Strap alpha layers for image registration
    alpha_src = img_src[:, :, 3]
    img_src = img_src[:, :, :3]
    alpha_dest = img_dest[:, :, 3]
    img_dest = img_dest[:, :, :3]

    img_src_g = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_dest_g = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)

    # Detect and Match
    mstart = time.time()
    matches, kp_src, kp_dest = func_dict[detector](img_src_g, img_dest_g)
    mduration = time.time() - mstart

    # Keypoint stats
    stats['src_keypoints'] = len(kp_src)
    stats['dest_keypoints'] = len(kp_dest)
    # Check if source kpts are in destination FOV
    pts_src = cv2.KeyPoint_convert(kp_src).astype(int)
    roi_mask = cv2.warpPerspective(alpha_dest, np.linalg.inv(H_gt), dsize)
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
    pts_src = cv2.KeyPoint_convert(kp_src, matches_qidx)
    pts_dest = cv2.KeyPoint_convert(kp_dest, matches_tidx)
    pts_proj_gt = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
    dists = np.linalg.norm(pts_proj_gt - pts_dest, axis=1) 
    match_mask = (dists <= match_threshold)
    stats['match_threshold'] = match_threshold
    stats['correct_matches'] = np.count_nonzero(match_mask)

    # Early abort
    if stats['correct_matches'] <= 4:
        return stats

    if verbose:
        print(f"Matched {len(matches)} from {len(kp_src)} source keypoints to {len(kp_dest)} destination keypoints in {mduration:03f}s")
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
    roi = 255*np.ones_like(img_src_g)
    roi_proj_gt = cv2.warpPerspective(roi, H_gt, (w, h))
    roi_proj_est = cv2.warpPerspective(roi, H, (w, h))
    iou = np.count_nonzero(np.bitwise_and(roi_proj_gt, roi_proj_est))/np.count_nonzero(np.bitwise_or(roi_proj_gt, roi_proj_est))
    stats['iou'] = iou

    if verbose:
        print(f"RANSAC inlier ratio: {np.count_nonzero(inlier_mask)/inlier_mask.shape[0]}")
        print(f"Estimated homography with {detector.upper()} in {hduration:03f}s")
        print(H)

        # Re-append alpha layers for cavity free composition
        img_src = np.concatenate((img_src, alpha_src[..., None]), axis=2)
        img_dest = np.concatenate((img_dest, alpha_dest[..., None]), axis=2)

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
    parser.add_argument('img_name', help="Path to src-image)")
    parser.add_argument('-d', '--detector', choices=func_dict.keys(), required=True, help="Determine which feature detector to use")
    parser.add_argument('-r', '--ratio', default=2.0, type=float, help="Ratio between src and dest image")
    parser.add_argument('-t', '--top', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Load files
    img = cv2.imread(args.img_name, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(-1, "Could not open file.", args.img_name)

    # Setup parameters
    match_threshold = 4.0
    ransac_params = dict(method=cv2.USAC_PROSAC,
                         ransacReprojThreshold=4.0,
                         maxIters=10000,
                         confidence=0.999999)

    stats = stitcher(img, args.detector, args.ratio, 
                     match_threshold, 
                     ransac_params,
                     top=args.top, verbose=args.verbose)

    print(stats)


