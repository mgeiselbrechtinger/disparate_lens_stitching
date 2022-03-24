import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

SCREEN_RES = (1920, 1080)

def main():
    # Handle arguments
    if len(sys.argv) != 3:
        print(f"USAGE: ./{sys.argv[0]} path/to/image[n1].jpg path/to/image[n2].jpg\n\tComputes homography between image[n1].jpg and image[n2].jpg\n\t,stores it in same directory as homography[n1]_[n2].csv") 
        sys.exit(-1)

    current_path = Path(__file__).parent

    img1_name = f"{current_path}/{sys.argv[1]}"
    img2_name = f"{current_path}/{sys.argv[2]}"
    img1_num = int(sys.argv[1].split('/')[-1][5:8])
    img2_num = int(sys.argv[2].split('/')[-1][5:8])
    res_path = '/'.join(img1_name.split('/')[:-1]) + f"/homography{img1_num:03d}_{img2_num:03d}.csv"

    # Load files
    img1 = cv2.imread(img1_name, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_name, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print("Couldn't load images")
        sys.exit(-1)

    # Generate selection window and record manual correspondences
    cv2.namedWindow("select_win", cv2.WINDOW_NORMAL)        
    cv2.setWindowProperty("select_win", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    img_comb = np.concatenate((img1, img2), axis=1)
    img1_pts = list()
    img2_pts = list()
    cv2.setMouseCallback('select_win', onMouse, [img_comb, img1_pts, img2_pts, img1.shape[1]])
    print("Click at list 4 corresponding points in both pictures alternating")
    while True:
        cv2.imshow("select_win", img_comb)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Estimate Homography from manual corresponcences
    H, _ = cv2.findHomography(np.array(img1_pts), np.array(img2_pts), 0)
    print(H)

    # Refinde Homography by ECC minimization
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 500, 1E-8)
    _, H_r = cv2.findTransformECC(img1_gray, img2_gray, np.array(H, dtype=np.float32), cv2.MOTION_HOMOGRAPHY, term_criteria)
    print(H_r)
    np.savetxt(res_path, H_r, delimiter=',')

    # Display and store final result
    # TODO transform corners to get resulting image size
    #img1_corners = np.array([[0, 0], [img1.shape[1]-1, 0], [0, img1.shape[0]-1], [img1.shape[1]-1, img1.shape[0]-1]], dtype=np.float32)
    #img1_warp_corners = cv2.perspectiveTransform(img1_corners, np.array(H_r, dtype=np.float32))
    #img1_warp_size = (int(np.amax(img1_warp_corners[:,0])), int(np.amax(img1_warp_corners[:,1])))
    img1_warp_size = (int(1.5*img1.shape[1]), int(1.5*img1.shape[0]))
    img1_warp = cv2.warpPerspective(img1, H_r, img1_warp_size)
    cv2.imshow("img1_warp_win", img1_warp)
    cv2.imshow("img2_win", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
