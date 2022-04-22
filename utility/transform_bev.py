################################################################################
#
#   Bird Eye View Transform
#
#   Has different modes of transformation
#   Assumes quadrangle is smaller on upper part if not comment-out 
#   source point sorting and mark points in appropriate order.
#
#
################################################################################

import argparse

import cv2
import numpy as np

# Modes: 
#   (0) Squash and strech lower and upper points respectively (90degree, information loss)
#   (1) Same as (0) but only half way (45degree)
#   (2) Stretch upper points (90degree, no information loss)
#   (3) Same as (2) but only half way (45degree)
#   (4) Squash lower points (90degree, max information loss)
#   (5) Same as (4) but only half way (45degree)
#   (6) Same as (0) but only quarter way (25degree)
mode_offsets = [lambda uw, lw: (-(lw - uw)/4, (lw - uw)/4, -(lw - uw)/4, (lw - uw)/4),
                lambda uw, lw: (-uw/2 - (lw-uw)/8, uw/2 + (lw-uw)/8, -lw/2 + (lw-uw)/8, lw/2 - (lw-uw)/8),
                lambda uw, lw: (-lw/2, lw/2, -lw/2, lw/2),
                lambda uw, lw: (-lw/4, lw/4, -lw/2, lw/2),
                lambda uw, lw: (-uw/2, uw/2, -uw/2, uw/2),
                lambda uw, lw: (-uw/2, uw/2, -uw, uw),
                lambda uw, lw: (-uw/2 - (lw-uw)/16, uw/2 + (lw-uw)/16, -lw/2 + (lw-uw)/16, lw/2 - (lw-uw)/16)]

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_names', nargs='+', help="Paths to image)")
    parser.add_argument('-w', '--write', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-m', '--mode', type=int, default=0, help="Transformation mode")
    args = parser.parse_args()

    for name in args.img_names:
        # Load files
        img = cv2.imread(name, cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(-1, "Could not open file.", name)

        h, w, _ = img.shape
        
        src_pts = manualCorrespondence(img)
        # Sort points into order: [[upper_left, upper_right, lower_left, lower_right]]
        src_pts = sorted(src_pts, key=lambda k: k[1]) # sort fist by y coordinate
        src_pts[:2] = sorted(src_pts[:2]) # sort upper by x coordinate
        src_pts[2:] = sorted(src_pts[2:]) # sort lower by x coordinate
        src_pts = np.float32(src_pts) 

        # Squash lower points and expand upper horizontal points to get rectangle
        uw = np.linalg.norm(src_pts[1] - src_pts[0]) 
        lw = np.linalg.norm(src_pts[3] - src_pts[2]) 
        dw = lw - uw 
        sl = np.linalg.norm(src_pts[2] - src_pts[0])*np.linalg.norm(src_pts[3] - src_pts[1])
        #dh = np.sqrt(sl - dw**2/4)
        dh = 3*h/4

        # Form destination points at lower center
        offset = mode_offsets[args.mode](uw, lw)
        dest_ul = np.float32([w/2 + offset[0], h - dh])
        dest_ur = np.float32([w/2 + offset[1], h - dh])
        dest_ll = np.float32([w/2 + offset[2], h])
        dest_lr = np.float32([w/2 + offset[3], h])
        dest_pts = np.stack([dest_ul, dest_ur, dest_ll, dest_lr], axis=0)

        H = cv2.getPerspectiveTransform(src_pts, dest_pts)

        img_bev = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC)

        if args.verbose:
            print(H)
            cv2.namedWindow("bird eye transform", cv2.WINDOW_NORMAL)        
            img_viz = np.concatenate([img, img_bev], axis=1)
            cv2.imshow('bird eye transform', img_viz)
            cv2.waitKey(0)

        if args.write:
            s = name.split('.')
            pre = '.'.join(s[:-1]) # Strip file ending
            #cv2.imwrite(f"{pre}_bev_{args.mode}.png", img_bev)
            np.savetxt(f"{pre}_bev_{args.mode}.csv", H, delimiter=',')


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

