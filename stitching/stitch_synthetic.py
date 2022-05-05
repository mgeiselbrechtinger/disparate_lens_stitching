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

stats = { 'keypoints'   : 0,
          'matches'     : 0,
          'ratio'       : 1 }

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name', help="Paths to src-image)")
    parser.add_argument('hg_names', nargs='*', default=[], help="Paths to src- and dest-transformation)")
    parser.add_argument('-d', '--detector', choices=func_dict.keys(), required=True, help="Determine which feature detector to use")
    parser.add_argument('-r', '--ratio', default=2.0, type=float, help="Ratio between src and dest image")
    parser.add_argument('-t', '--top', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-o', '--outfile', help="Path for stitched image")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-b', '--benchmark', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.benchmark:
        args.verbose = False

    # Load files
    img = cv2.imread(args.img_name, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(-1, "Could not open file.", args.img_name)

    # Transform images with alpha layers for easy composition
    h, w, _ = img.shape
    r = 1/args.ratio
    stats['ratio'] = r
    dsize = int(w*r), int(h*r)
    if len(args.hg_names) == 0:
        H_src = np.eye(3)
    else:
        H_src = np.loadtxt(args.hg_names[0], delimiter=',')
    H_dest = np.copy(H_src)

    # Downscale source image to mimic higher resolution of destination image
    H_src[0, 2] *= r 
    H_src[1, 2] *= r 
    H_src[2, 0] /= r 
    H_src[2, 1] /= r 
    img_src = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
    img_src = np.concatenate((img_src, 255*np.ones_like(img_src[..., :1])), axis=2)
    #img_src = cv2.warpPerspective(img_src, H_src, dsize, flags=cv2.INTER_LINEAR)
    S = np.float32([[1/r, 0, 0], [0, 1/r, 0], [0, 0, 1]])
    H_src_inv = S.dot(np.linalg.inv(H_src))

    # Crop destination image to mimic restricted FOV
    img_dest = (img*0.75).astype(np.uint8)
    img_dest = img_dest[int(h - h*r)//2 : int(h + h*r)//2, int(w - w*r)//2 : int(w + w*r)//2]
    img_dest = np.concatenate((img_dest, 255*np.ones_like(img_dest[..., :1])), axis=2)
    A = np.float32([[1, 0, (w - w*r)/2], [0, 1, (h - h*r)/2], [0, 0, 1]])
    if len(args.hg_names) == 2:
        # Use dedicated homography for cropped image
        H_dest = np.loadtxt(args.hg_names[1], delimiter=',')
        #img_dest = cv2.warpPerspective(img_dest, H_dest, dsize, flags=cv2.INTER_LINEAR)
        H_gt = H_dest.dot(np.linalg.inv(A).dot(H_src_inv))
    else:
        # Shift and warp with full image homography and crop again
        #img_dest = cv2.warpPerspective(img_dest, H_dest.dot(A), (w, h), flags=cv2.INTER_LINEAR)
        #img_dest = img_dest[int(h - h*r)//2 : int(h + h*r)//2, int(w - w*r)//2 : int(w + w*r)//2]
        H_gt = np.linalg.inv(A).dot(H_dest.dot(H_src_inv))
    
    # Rescale groundtruth homography
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
    matches, kp_src, kp_dest = func_dict[args.detector](img_src_g, img_dest_g)
    mduration = time.time() - mstart

    # Check keypoint stats
    n_kp_dest = len(kp_dest)
    # Check if source kpts are in destination FOV
    pts_src = cv2.KeyPoint_convert(kp_src).astype(int)
    roi_mask = cv2.warpPerspective(alpha_dest, np.linalg.inv(H_gt), dsize)
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

    # Find inliers using ground truth
    inlier_threshold = 4
    pts_src = cv2.KeyPoint_convert(kp_src, matches_qidx)
    pts_dest = cv2.KeyPoint_convert(kp_dest, matches_tidx)
    pts_src_ref = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
    dists = np.linalg.norm(pts_src_ref - pts_dest, axis=1) 
    inlier_mask = (dists < inlier_threshold)
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

    if args.benchmark:
        print(f"Stats: {stats}")
        return
    
    # Estimate Homography
    hstart = time.time()

    ransac_params = dict(method=cv2.USAC_MAGSAC,
                         ransacReprojThreshold=0.25,
                         maxIters=10000,
                         confidence=0.999999)

    # Sort keypoints for PROSAC
    if ransac_params['method'] == cv2.USAC_PROSAC:
        sort_args = matches_dist.argsort()
        pts_src = pts_src[sort_args]
        pts_dest = pts_dest[sort_args]

    H, inlier_mask = cv2.findHomography(pts_src, pts_dest, **ransac_params)
    hduration = time.time() - hstart

    if H is None:
        raise ValueError("Homography estimation failed")

    if args.verbose:
        print(f"RANSAC inlier ratio: {np.count_nonzero(inlier_mask)/inlier_mask.shape[0]}")
        print(f"Estimated homography with {args.detector.upper()} in {hduration:03f}s")
        print(H)
        print("Elementwise error")
        print(np.abs(H_gt - H))
        print(np.allclose(H_gt, H, 0.2, 0.1))

        # Re-append alpha layers for cavity free composition
        img_src = np.concatenate((img_src, alpha_src[..., None]), axis=2)
        img_dest = np.concatenate((img_dest, alpha_dest[..., None]), axis=2)

        img_dest = cv2.warpPerspective(img_dest, H_dest, dsize, flags=cv2.INTER_LINEAR)
        res = compose(img_dest, [img_src], [H_dest.dot(H)], base_on_top=args.top)

        cv2.namedWindow("composition", cv2.WINDOW_NORMAL)        
        cv2.imshow("composition", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Done in {mduration+hduration:03f}s")
        print(f"Scale change h, w: {res.shape[0]/img_src.shape[0], res.shape[1]/img_src.shape[1]}")


if __name__ == '__main__':
    main()

