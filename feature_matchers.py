################################################################################
#
#   Feature matching library
#
#   TODO use dictionary to get function from 
#   TODO allow different matching algorithms
#       BFMatcher.knnMatch(des1, des2, k=2) to implement Lowes ratio test
#
################################################################################

import cv2
import numpy as np


def orb_detect_and_match(ref_img, mod_img):
    detector = cv2.ORB_create()
    # get keypoints and descriptors
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    # Use hamming distance for binary descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors using bruteforce
    matches = matcher.match(queryDescriptors=ref_des, 
                            trainDescriptors=mod_des)

    return matches, ref_kp, mod_kp

def brisk_detect_and_match(ref_img, mod_img):
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

def akaze_detect_and_match(ref_img, mod_img):
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

def sift_detect_and_match(ref_img, mod_img):
    detector = cv2.SIFT_create()
    # Get keypoints and descriptors
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    # RootSIFT version 
    # TODO make available as option
    #ref_des = np.sqrt(ref_des/np.linalg.norm(ref_des, ord=1, axis=0))
    #mod_des = np.sqrt(mod_des/np.linalg.norm(mod_des, ord=1, axis=0))

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors using bruteforce
    matches = matcher.match(queryDescriptors=ref_des, 
                            trainDescriptors=mod_des)

    return matches, ref_kp, mod_kp

def mix_feature_matching(ref_img, mod_img):
    # TODO allow different combinations
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

