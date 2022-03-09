from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def brisk_feature_matching(ref_img, mod_img):
    detector = cv2.BRISK_create()
    # Get keypoints and descriptors
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    # Use hamming distance for binary descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors using bruteforce
    matches = matcher.match(queryDescriptors=ref_des, 
                            trainDescriptors=mod_des)

    return matches, ref_kp, mod_kp

def akaze_feature_matching(ref_img, mod_img):
    detector = cv2.AKAZE_create()
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

def mix_feature_matching(ref_img, mod_img):
    orb = cv2.ORB_create()
    ref_kp = orb.detect(ref_img, None)
    mod_kp = orb.detect(mod_img, None)

    brisk = cv2.BRISK_create()
    ref_kp, ref_des = brisk.compute(ref_img, ref_kp) 
    mod_kp, mod_des = brisk.compute(mod_img, mod_kp)    

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(queryDescriptors=ref_des, 
                            trainDescriptors=mod_des)

    return matches, ref_kp, mod_kp

if __name__ == '__main__':
    current_path = Path(__file__).parent
    img_num = 0
    img_name = f"{current_path}/data/pattern/image{img_num:03d}.jpg"
    ref_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError("Couldn't load image: " + img_name)

    # Transform image
    scale = 2 #1/np.sqrt(2) # unitless but multiples of sqrt(2) are good
    rotation = 0 # degree
    translation = np.array([0, 0]) # pixels

    img_size = ref_img.shape[1], ref_img.shape[0]
    img_center = [(i-1)/2.0 for i in img_size]
    A = cv2.getRotationMatrix2D(img_center, rotation, scale)
    A[:,2] += translation
    mod_img = cv2.warpAffine(ref_img, A, img_size, borderMode=cv2.BORDER_REPLICATE)
    #mod_img = mod_img[img_size[0]//4 : 3*img_size[0]//4 + 1, 
    #                  img_size[1]//4 : 3*img_size[1]//4 + 1]
    #ref_img = ref_img[1*img_size[1]//8 : 7*img_size[1]//8 + 1, 
    #                  img_size[0]//2 : ]
    
    print(mod_img.shape)
    
    #cv2.imshow('Transformation', mod_img)

    matches, ref_kp, mod_kp = brisk_feature_matching(ref_img, mod_img)

    draw_params = dict(flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS |
                             cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    matches_img = cv2.drawMatches(ref_img, ref_kp,
                                  mod_img, mod_kp,
                                  matches[::5], None, **draw_params)

    cv2.imshow('matches', matches_img)

    # TODO compute affine tf matrix given matches and compare with original tf
    # Transform ref img kp and check if in vicinity of matched mod img kp
    ref_pts = cv2.KeyPoint.convert(ref_kp)
    mod_pts = cv2.KeyPoint.convert(mod_kp)
    ref_pts_h = np.c_[ref_pts, np.ones((ref_pts.shape[0], 1))]
    ref_pts_tf = ref_pts_h.dot(A.T)
    # TODO sort kp with index list from matches to avoid loop
    vicinity_th = 4
    pos_cnt = 0
    for m in matches:
        pt_m = mod_pts[m.trainIdx]
        pt_r = ref_pts[m.queryIdx]
        pt_r2m = ref_pts_tf[m.queryIdx]
        if np.all(np.abs(pt_r2m - pt_m) <= vicinity_th):
            pos_cnt += 1

    print(f"{pos_cnt}/{len(matches)} in vicinity of {vicinity_th} pixels")
    print(f"inlier ratio = {pos_cnt/len(matches)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def knn_matching():
    # TODO check params
    index_params= dict(algorithm=6,
                        table_number=6, # 12 
                        key_size = 12, # 20 
                        multi_probe_level = 1) #2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(ref_des, mod_des, k=3)
    # Only show every fifth match, otherwise it gets to overwhelming
    match_mask = np.zeros(np.array(matches).shape, dtype=int)
    match_mask[::5, ...] = 1

    draw_params = dict(matchesMask=match_mask,
                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    matches_img = cv2.drawMatchesKnn(ref_img,
                                     ref_kp,
                                     mod_img,
                                     mod_kp,
                                     matches,
                                     None,
                                     **draw_params)
    plt.figure()
    plt.imshow(matches_img)
    plt.show()
