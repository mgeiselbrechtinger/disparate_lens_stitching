################################################################################
#
#   Bird Eye View Example
#
#
################################################################################

import argparse

import cv2
import numpy as np

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_names', nargs='+', help="Paths to src- and dest-image)")
    parser.add_argument('-o', '--outfile', help="Path for transformed image")
    parser.add_argument('-l', '--load', help="Path to transform homography")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    imgs = list()
    imgs_warped = list()
    hgs = list()

    for name in args.img_names:
        # Load files
        img_full = cv2.imread(name, cv2.IMREAD_COLOR)
        if img_full is None:
            raise OSError(-1, "Could not open file.", name)

        img = img_full #img_full[img_full.shape[0]//2:]
        imgs.append(img)
        
        if args.load is None:
            src_pts = manualCorrespondence(img)
            src_pts = sorted(src_pts, key=lambda k: k[1]) # sort fist by y coordinate
            src_pts[:2] = sorted(src_pts[:2]) # sort upper by x coordinate
            src_pts[2:] = sorted(src_pts[2:]) # sort lower by x coordinate
            src_pts = np.float32(src_pts) # [[upper_left, upper_right, lower_left, lower_right]]
            # Squash lower horizontal points to match upper ones
            # For height take geometric mean
            height = np.sqrt((src_pts[2, 1] - src_pts[0, 1])*(src_pts[3, 1] - src_pts[1, 1]))
            dest_pts = np.concatenate([src_pts[:2], np.float32([[src_pts[0, 0], src_pts[0, 1]+height]]),
                np.float32([[src_pts[1, 0], src_pts[1, 1]+height]])], axis=0)

            H = cv2.getPerspectiveTransform(src_pts, dest_pts)

        else:
            H = np.loadtxt(args.load)

        hgs.append(H)
        # Shift projected image down to increase FOV
        t_y = 0 #img.shape[0]
        A = np.float32([[1, 0, 0], [0, 1, t_y], [0, 0, 1]])
        AH = A.dot(H)
        img_bev = cv2.warpPerspective(img, AH, (img_full.shape[1], img_full.shape[0]))
        imgs_warped.append(img_bev)

    if args.verbose:
        for H in hgs: print(H)
        #np.savetxt("../data/tu_stitching/seq2/bev/c39_0.csv", H)
        cv2.namedWindow("bird eye transform", cv2.WINDOW_NORMAL)        
        img_viz = np.concatenate(imgs_warped, axis=1)
        cv2.imshow('bird eye transform', img_viz)
        cv2.waitKey(0)

    if not args.outfile is None:
        # Strip file ending
        s = args.outfile.split('.')
        pre = ".".join(s[:-1])
        for i, img in enumerate(imgs_warped):
            cv2.imwrite(f"{pre}_{i:03d}.png", img)


def manualCorrespondence(img):
    img = np.copy(img)
    cv2.namedWindow("select_win", cv2.WINDOW_NORMAL)        
    cv2.setWindowProperty("select_win", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    src_pts = list()
    cv2.setMouseCallback('select_win', onMouse, [img, src_pts])
    print("Mark 4 points forming a rectangle in bird eye view, exit with <q>")
    while True:
        cv2.imshow("select_win", img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return src_pts


def onMouse(event, x, y, flags, param):
    # Store and mark coordinates on mouse click
    # remove last entry on middle click
    img = param[0]
    pts = param[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 4, color=(0,0,255), thickness=2)
        pts.append((x, y))

    elif False and event == cv2.EVENT_MBUTTONDOWN:
        pt = pts1.pop()
        cv2.circle(img, pt, 4, color=(255,255,255), thickness=3)

if __name__ == '__main__':
    main()

