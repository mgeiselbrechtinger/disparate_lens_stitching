import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

posList = []
def onMouse(event, x, y, flags, param):
    l = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        l.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        _ = l.pop()

def main():
    if len(sys.argv) != 3:
        print(f"USAGE: ./{sys.argv[0]} path/to/image[n1].jpg path/to/image[n2].jpg\n\tComputes homography between image[n1].jpg and image[n2].jpg\n\t,stores it in same directory as homography[n1]_[n2].csv") 
        sys.exit(-1)

    current_path = Path(__file__).parent

    img1_name = f"{current_path}/{sys.argv[1]}"
    img2_name = f"{current_path}/{sys.argv[2]}"
    res_path = '/'.join(img1_name.split('/')[:-1]) + '/'
    img1_num = int(sys.argv[1].split('/')[-1][5:8])
    img2_num = int(sys.argv[2].split('/')[-1][5:8])
    print(img1_num)
    print(img2_num)
    print(res_path)
    img1 = cv2.imread(img1_name, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_name, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print("Couldn't load images")
        sys.exit(-1)

    img1_pts = list()
    img2_pts = list()
    cv2.imshow("img1_win", img1)
    cv2.imshow("img2_win", img2)

    cv2.setMouseCallback('img1_win', onMouse, [img1_pts])
    cv2.setMouseCallback('img2_win', onMouse, [img2_pts])
    print("Click at list 4 corresponding points in both pictures")
    # TODO mark points
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    H, _ = cv2.findHomography(np.array(img1_pts), np.array(img2_pts), 0)
    print(H)

    # TODO refine estimated homography with lukas-kanade algo
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1E-8)
    _, H_r = cv2.findTransformECC(img1_gray, img2_gray, H.astype(np.float32), cv2.MOTION_HOMOGRAPHY, term_criteria)

    # TODO store final homography in csv file
    print(H)

    img1_size = img1.shape[1], img1.shape[0]
    img1_warp = cv2.warpPerspective(img1, H_r, img1_size)
    cv2.imshow("img1_warp_win", img1_warp)
    cv2.imshow("img2_win", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
