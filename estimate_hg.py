import sys
from pathlib import Path

import cv2
import numpy as np

ALGO = "orb" # "orb" | "sift"

def main():
    # Handle arguments
    if len(sys.argv) != 3:
        print(f"USAGE: ./{sys.argv[0]} path/to/image[n1].jpg path/to/image[n2].jpg\n\tComputes homography between image[n1].jpg and image[n2].jpg\n\t,stores it in same directory as homography_{ALGO}[n1]_[n2].csv") 
        sys.exit(-1)

    current_path = Path(__file__).parent

    img1_name = f"{current_path}/{sys.argv[1]}"
    img2_name = f"{current_path}/{sys.argv[2]}"
    img1_num = int(sys.argv[1].split('/')[-1][5:8])
    img2_num = int(sys.argv[2].split('/')[-1][5:8])
    ref_path = '/'.join(img1_name.split('/')[:-1]) + f"/homography{img1_num:03d}_{img2_num:03d}.csv"
    res_path = '/'.join(img1_name.split('/')[:-1]) + f"/{ALGO}{img1_num:03d}_{img2_num:03d}.csv"

    # Load files
    img1 = cv2.imread(img1_name, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_name, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print("Couldn't load images")
        sys.exit(-1)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Matching
    if ALGO == "sift":
        matches, kp1, kp2 = sift_feature_matching(img1_gray, img2_gray)
    elif ALGO == "orb":
        matches, kp1, kp2 = orb_feature_matching(img1_gray, img2_gray)
    else:
        print("No such algorithm " + ALGO)
        sys.exit(-1)

    # TODO implement Lowe ratio test
    img1_pts = list()
    img2_pts = list()
    for m in matches:
        img1_pts.append(kp1[m.queryIdx].pt)
        img2_pts.append(kp2[m.trainIdx].pt)

    H, inlier_mask = cv2.findHomography(np.array(img1_pts, dtype=np.float32), 
                                        np.array(img2_pts, dtype=np.float32), 
                                        cv2.RANSAC, 5.0)
    
    H_ref = np.loadtxt(ref_path, delimiter=',')
    print(H)
    print(H_ref)
    norm = np.linalg.norm(H - H_ref)
    print(norm)

    np.savetxt(res_path, H, delimiter=',')


def orb_feature_matching(ref_img, mod_img):
    detector = cv2.ORB_create()
    # Get keypoints and descriptors
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    # Use hamming distance for binary descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors using bruteforce
    matches = matcher.match(queryDescriptors=ref_des, 
                            trainDescriptors=mod_des)

    return matches, ref_kp, mod_kp


def sift_feature_matching(ref_img, mod_img):
    detector = cv2.SIFT_create()
    # Get keypoints and descriptors
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors using bruteforce
    matches = matcher.match(queryDescriptors=ref_des, 
                            trainDescriptors=mod_des)

    return matches, ref_kp, mod_kp

if __name__ == '__main__':
    main()

