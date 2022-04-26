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

# Modes: TODO implement
#   mixed: distribute IPM over upper and lower points 
#   upper: stretch upper points to fit IPM
#   lower: squash lower points to fit IPM
MODE = 'mixed'

# Steps:
#   number of IPMs to calculate between 90 to 10 degrees
STEPS = 10

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name', help="Path to image")
    parser.add_argument('-o', '--output', nargs='?', help="Output directory")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    #parser.add_argument('-w', '--write', action=argparse.BooleanOptionalAction, default=False)
    #parser.add_argument('-m', '--mode', type=int, default=0, help="Transformation mode")
    args = parser.parse_args()

    img = cv2.imread(args.img_name, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(-1, "Could not open file.", args.img_name)

    h, w, _ = img.shape
    
    src_pts = manualCorrespondence(img)
    # Sort points into order: [[upper_left, upper_right, lower_left, lower_right]]
    src_pts = sorted(src_pts, key=lambda k: k[1]) # sort fist by y coordinate
    src_pts[:2] = sorted(src_pts[:2]) # sort upper by x coordinate
    src_pts[2:] = sorted(src_pts[2:]) # sort lower by x coordinate
    src_pts = np.float32(src_pts) 

    # Parameters for destination rectangle
    uw = np.linalg.norm(src_pts[1] - src_pts[0]) 
    lw = np.linalg.norm(src_pts[3] - src_pts[2]) 
    diff_width = lw - uw 
    #dist = np.sqrt(np.linalg.norm(src_pts[2] - src_pts[0])*np.linalg.norm(src_pts[3] - src_pts[1]))
    dist = np.sqrt(src_pts[0, 1]*src_pts[1, 1])


    # linearly increase distortion 
    width_incs = np.linspace(0, diff_width, STEPS)
    height_incs = np.linspace(dist, h, STEPS)

    for step in range(STEPS): 
        dw = width_incs[step]
        dh = height_incs[step]
        dest_ul = np.float32([w/2 - uw/2 - dw/4, h - dh])
        dest_ur = np.float32([w/2 + uw/2 + dw/4, h - dh])
        dest_ll = np.float32([w/2 - lw/2 + dw/4, h])
        dest_lr = np.float32([w/2 + lw/2 - dw/4, h])
        dest_pts = np.stack([dest_ul, dest_ur, dest_ll, dest_lr], axis=0)

        H = cv2.getPerspectiveTransform(src_pts, dest_pts)

        if args.verbose:
            print(H)
            img_bev = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC)
            cv2.namedWindow("bird eye transform", cv2.WINDOW_NORMAL)        
            img_viz = np.concatenate([img, img_bev], axis=1)
            cv2.imshow('bird eye transform', img_viz)
            cv2.waitKey(0)

        if not args.output is None:
            np.savetxt(f"{args.output}/bev_{MODE}_{step}.csv", H, delimiter=',')



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

