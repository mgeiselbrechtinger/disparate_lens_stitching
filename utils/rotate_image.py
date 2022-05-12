################################################################################
#
#   Bird Eye View Example
#
#
################################################################################

import argparse

import cv2
import numpy as np

STEPS = 10
MODE = "rotated"

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name', help="Path to image")
    parser.add_argument('-o', '--output', help="Path to output directory")
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-r', '--ratio', default=1.0, type=float, help="Ratio between src and dest image")
    args = parser.parse_args()

    img = cv2.imread(args.img_name, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(-1, "Could not open file.", args.img_name)

    # Crop image ratio from center
    r = 1/args.ratio
    h, w, _ = img.shape
    img = img[int(h - h*r)//2 : int(h + h*r)//2, int(w - w*r)//2 : int(w + w*r)//2]

    h, w, _ = img.shape

    alphas = np.linspace(0, 8.0E-5, STEPS)
    scales = np.linspace(1, 8*r, STEPS)
    for step in range(STEPS):
        alpha = alphas[step]
        s_x, s_y = 1, scales[step]
        d = 1
        f = 3.8E-3

        K_inv = np.float32([[1/f, 0, -1/f*w/2], 
                            [0, 1/f, -1/f*h/2],
                            [0, 0, 0],
                            [0, 0, 1]])

        R = np.eye(4)
        rvec = alpha*np.float32([[1, 0, 0]])
        R[:3, :3], _ = cv2.Rodrigues(rvec)

        S = np.float32([[s_x, 0, 0, 0],
                        [0, s_y, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        T = np.float32([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, d],
                        [0, 0, 0, 1]])

        K = np.float32([[f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        H = K.dot(T.dot(S.dot(R.dot(K_inv))))

        # Move image down to fill free space on bottom after rotation
        p = np.float32([[w/2, h]]).reshape(-1, 1, 2)
        p_h = cv2.perspectiveTransform(p, H)
        A = np.float32([[1, 0, -(p_h[0, 0, 0] - p[0, 0, 0])], [0, 1, -(p_h[0, 0, 1] - p[0, 0, 1])], [0, 0, 1]])
        AH = A.dot(H)

        if args.verbose:
            print(AH)
            img_bev = cv2.warpPerspective(img, AH, (w, h))
            cv2.namedWindow("bird eye transform", cv2.WINDOW_NORMAL)        
            cv2.imshow('bird eye transform', img_bev)
            cv2.waitKey(0)

        if not args.output is None:
            np.savetxt(f"{args.output}/bev_{MODE}_{step}.csv", AH, delimiter=',')


if __name__ == '__main__':
    main()

