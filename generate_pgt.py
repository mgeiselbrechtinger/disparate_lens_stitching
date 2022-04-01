################################################################################
#
#   Construct grount truth homography from src to dest image by refinement
#       of hand labeled correspondences
#
#
################################################################################

import argparse
from pathlib import Path

import cv2
import numpy as np

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_names', nargs=2, help="Paths to src- and dest-image)")
    parser.add_argument('-o', '--outfile', dest='hg_name', help="Path for homography csv-file")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Load files
    img_src = cv2.imread(args.img_names[0], cv2.IMREAD_COLOR)
    img_dest = cv2.imread(args.img_names[1], cv2.IMREAD_COLOR)
    if img_src is None or img_dest is None:
        raise OSError(-1, "Could not open file.", args.img_names[0], args.img_names[1])

    img_src_pts, img_dest_pts = manualCorrespondence(img_src, img_dest)
    if len(img_src_pts) != len(img_dest_pts) or len(img_src_pts) < 4:
        raise Exception("Invalid correspondences")

    # Estimate Homography from manual corresponcences using LS
    H, _ = cv2.findHomography(np.array(img_src_pts), np.array(img_dest_pts), 0)
    if args.verbose:
        print("Homography estimated from hand picked correspondences")
        print(H)

    # Refinde Homography by ECC minimization
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_dest_gray = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 150, 1E-6)
    _, H_r = cv2.findTransformECC(img_src_gray, img_dest_gray, 
                                  np.array(H, dtype=np.float32), 
                                  cv2.MOTION_HOMOGRAPHY, term_criteria)
    if args.verbose:
        print("Final pseudo ground truth homography")
        print(H_r)

    if args.hg_name != None:
        np.savetxt(args.hg_name, H_r, delimiter=',')


def manualCorrespondence(img1, img2):
    # Generate selection window and record manual correspondences
    cv2.namedWindow("select_win", cv2.WINDOW_NORMAL)        
    cv2.setWindowProperty("select_win", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    img_comb = np.concatenate((img1, img2), axis=1)
    img1_pts = list()
    img2_pts = list()
    cv2.setMouseCallback('select_win', onMouse, [img_comb, img1_pts, img2_pts, img1.shape[1]])
    print("Mark at least 4 corresponding points in both pictures, exit with <q>")
    while True:
        cv2.imshow("select_win", img_comb)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return img1_pts, img2_pts


def onMouse(event, x, y, flags, param):
    # Store and mark coordinates on mouse click
    # remove last entry on middle click
    img = param[0]
    pts1 = param[1]
    pts2 = param[2]
    img1_width = param[3]
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < img1_width:
            cv2.circle(img, (x, y), 4, color=min(255, 30*len(pts1)), thickness=1)
            pts1.append((x, y))

        else:
            cv2.circle(img, (x, y), 4, color=min(255, 30*len(pts2)), thickness=1)
            pts2.append((x-img1_width, y))

    elif event == cv2.EVENT_MBUTTONDOWN:
        pt = tuple()
        if x < img1_width:
            pt = pts1.pop()

        else:
            pt = pts2.pop()
            pt = (pt[0]+img1_width, pt[1])

        cv2.circle(img, pt, 4, color=(255,255,255), thickness=2)


if __name__=="__main__":
    main()
