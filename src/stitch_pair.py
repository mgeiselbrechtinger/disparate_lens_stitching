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
    parser.add_argument('img_names', nargs=2, help="Paths to src- and dest-image)")
    parser.add_argument('-d', '--detector', choices=func_dict.keys(), required=True, help="Determine which feature detector to use")
    parser.add_argument('-t', '--top', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-o', '--outfile', help="Path for stitched image")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-r', '--ratio', type=float, default=1.0, help="Scale ratio for images")
    parser.add_argument('-w', '--warp', nargs=2, help="Paths to src- and dest-image warps)")
    args = parser.parse_args()

    # Load files
    img_src = cv2.imread(args.img_names[0], cv2.IMREAD_COLOR)
    img_dest = cv2.imread(args.img_names[1], cv2.IMREAD_COLOR)
    if img_src is None or img_dest is None:
        raise OSError(-1, "Could not open file.", args.img_names[0], args.img_names[1])

    if not args.warp is None:
        H_src = np.loadtxt(args.warp[0], delimiter=',')
        H_dest = np.loadtxt(args.warp[1], delimiter=',')
        #img_src = cv2.warpPerspective(img_src, H_src, (int(img_src.shape[1]), int(img_src.shape[0])))
        #img_dest = cv2.warpPerspective(img_dest, H_dest, (int(img_dest.shape[1]), int(img_dest.shape[0])))

    img_src_b = img_src
    img_dest_b = img_dest

    # Down scale images
    r = 1/args.ratio

    h, w, _ = img_src.shape
    dsize = int(w*r), int(h*r)
    img_src = cv2.resize(img_src, dsize, interpolation=cv2.INTER_AREA)

    h, w, _ = img_dest.shape
    dsize = int(w*r), int(h*r)
    img_dest = cv2.resize(img_dest, dsize, interpolation=cv2.INTER_AREA)

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

        draw_params = dict(flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS |
                                 cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        matches_img = cv2.drawMatches(img_src, kp_src, img_dest, kp_dest,
                                      matches[::2], None, **draw_params)

        cv2.namedWindow("matches", cv2.WINDOW_NORMAL)        
        cv2.setWindowProperty("matches", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('matches', matches_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ransac_params = dict(method=cv2.USAC_PROSAC,
                         ransacReprojThreshold=1.25,
                         maxIters=10000,
                         confidence=0.999999)

    pts_src = cv2.KeyPoint_convert(kp_src, matches_qidx)
    pts_dest = cv2.KeyPoint_convert(kp_dest, matches_tidx)
    # Sort keypoints (only!) for PROSAC
    if ransac_params['method'] == cv2.USAC_PROSAC:
        sort_args = matches_dist.argsort()
        pts_src = pts_src[sort_args]
        pts_dest = pts_dest[sort_args]

    hstart = time.time()
    H, inlier_mask = cv2.findHomography(pts_src, pts_dest, **ransac_params)

    hduration = time.time() - hstart

    if H is None:
        raise ValueError("Homography estimation failed")

    # Up sale homography to fit original images 
    H[0, 2] /= r 
    H[1, 2] /= r 
    H[2, 0] *= r 
    H[2, 1] *= r 

    if args.verbose:
        print(f"Estimated homography with {args.detector.upper()} in {hduration:03f}s")
        print(H)

    img_src = img_src_b
    img_dest = img_dest_b

    if not args.warp is None:
        alpha_src = 255*np.ones_like(img_src[..., :1])
        img_src = np.concatenate((img_src, alpha_src), axis=2)
        #alpha_src = cv2.warpPerspective(alpha_src, H_src, (img_src.shape[1], img_src.shape[0]))
        alpha_dest = 255*np.ones_like(img_dest[..., :1])
        img_dest = np.concatenate((img_dest, alpha_dest), axis=2)
        img_dest = cv2.warpPerspective(img_dest, H_dest, (img_dest.shape[1], img_dest.shape[0]))
        H = H_dest.dot(H)

    res = compose(img_dest, [img_src], [H], base_on_top=args.top)

    cv2.namedWindow("composition", cv2.WINDOW_NORMAL)        
    cv2.imshow("composition", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not args.outfile is None:
        cv2.imwrite(args.outfile, res)

    print(f"Done in {mduration+hduration:03f}s")

    if args.verbose:
        print(f"Scale change h, w: {res.shape[0]/img_src.shape[0], res.shape[1]/img_src.shape[1]}")

if __name__ == '__main__':
    main()

