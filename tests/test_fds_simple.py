from pathlib import Path

import cv2
import numpy as np

###
# Mimic Camera setup:
#   reference image: smaller FOV, original resolution
#   modified image:  original FOV, lower resolution
def main():
    current_path = Path(__file__).parent
    img_num = 0
    img_name = f"{current_path}/data/cmu_s7_c0/image{img_num:03d}.jpg"
    ref_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError("Couldn't load image: " + img_name)

    # Transform image
    # TODO: adjust affine transform
    scale = 1 
    rotation = 0 # degree
    translation = np.array([0, 0]) # pixels

    img_size = ref_img.shape[1], ref_img.shape[0]
    img_center = [(i-1)/2.0 for i in img_size]
    A = cv2.getRotationMatrix2D(img_center, rotation, scale)
    A[:,2] += translation
    mod_img = cv2.warpAffine(ref_img, A, img_size, borderMode=cv2.BORDER_REPLICATE)
    # Change resolution
    # TODO: scale up after scaling down
    mod_img = cv2.pyrDown(mod_img) # 1:2
    #mod_img = cv2.pyrDown(mod_img) # 1:4
    #mod_img = cv2.pyrDown(mod_img) # 1:8

    #mod_img = cv2.pyrUp(mod_img
    #mod_img = cv2.pyrUp(mod_img)
    mod_img = cv2.pyrUp(mod_img)

    # Half FOV
    # TODO: choose half or quarter FOV
    fov_r = 0.25 # 0.375
    ref_img[: int(fov_r*img_size[1]), :] = 0
    ref_img[-int(fov_r*img_size[1]) :, :] = 0
    ref_img[:, : int(fov_r*img_size[0])] = 0
    ref_img[:, -int(fov_r*img_size[0]) :] = 0

    # Match and visualize
    # TODO: choose between SIFT and ORB
    matches, ref_kp, mod_kp = orb_feature_matching(ref_img, mod_img)
    #matches, ref_kp, mod_kp = sift_feature_matching(ref_img, mod_img)

    draw_params = dict(flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS |
                             cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    matches_img = cv2.drawMatches(ref_img, ref_kp,
                                  mod_img, mod_kp,
                                  matches[::2][:20], None, **draw_params)

    cv2.imshow('matches', matches_img)

    # Quick comparison of matching
    # Transform ref img kp and check if in vicinity of matched mod img kp
    ref_pts = cv2.KeyPoint.convert(ref_kp)
    mod_pts = cv2.KeyPoint.convert(mod_kp)
    ref_pts_h = np.c_[ref_pts, np.ones((ref_pts.shape[0], 1))]
    ref_pts_tf = ref_pts_h.dot(A.T)
    vicinity_th = 4
    pos_cnt = 0
    for m in matches:
        pt_m = mod_pts[m.trainIdx]
        pt_r = ref_pts[m.queryIdx]
        pt_r2m = ref_pts_tf[m.queryIdx]
        if np.all(np.abs(pt_r2m - pt_m) <= vicinity_th):
            pos_cnt += 1

    print(f"{pos_cnt}/{len(matches)} matches in vicinity of {vicinity_th} pixels")
    print(f"inlier ratio = {pos_cnt/len(matches)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

