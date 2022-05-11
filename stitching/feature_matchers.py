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
#import onnx
#import onnxruntime
#from extract_patches.core import extract_patches

ONNX_PATH = "./onnx_models/"

def orb_detect_and_match(ref_img, mod_img):
    detector = cv2.ORB_create()
    
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    matches = match(ref_kp, ref_des, mod_kp, mod_des, des_type=cv2.NORM_HAMMING)
    return matches, ref_kp, mod_kp

def brisk_detect_and_match(ref_img, mod_img):
    detector = cv2.BRISK_create()
    
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    matches = match(ref_kp, ref_des, mod_kp, mod_des, des_type=cv2.NORM_HAMMING)
    return matches, ref_kp, mod_kp

def akaze_detect_and_match(ref_img, mod_img):
    detector = cv2.AKAZE_create()
    
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    matches = match(ref_kp, ref_des, mod_kp, mod_des, des_type=cv2.NORM_HAMMING)
    return matches, ref_kp, mod_kp

def surf_detect_and_match(ref_img, mod_img):
    detector = cv2.xfeatures2d.SURF_create()
    
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    matches = match(ref_kp, ref_des, mod_kp, mod_des)
    return matches, ref_kp, mod_kp

def sift_detect_and_match(ref_img, mod_img):
    detector = cv2.SIFT_create()
   
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    # RootSIFT version 
    #ref_des = np.sqrt(ref_des)#/np.linalg.norm(ref_des, ord=1, axis=0))
    #mod_des = np.sqrt(mod_des)#/np.linalg.norm(mod_des, ord=1, axis=0))

    matches = match(ref_kp, ref_des, mod_kp, mod_des)
    return matches, ref_kp, mod_kp

def mix_detect_and_match(ref_img, mod_img):
    # SURF keypoints
    detector = cv2.xfeatures2d.SURF_create()
    ref_kp = detector.detect(ref_img, None)
    mod_kp = detector.detect(mod_img, None)

    # SIFT descriptors
    descriptor = cv2.SIFT_create()
    ref_kp, ref_des = descriptor.compute(ref_img, ref_kp)
    mod_kp, mod_des = descriptor.compute(mod_img, mod_kp)

    matches = match(ref_kp, ref_des, mod_kp, mod_des)
    return matches, ref_kp, mod_kp

def l2net_detect_and_match(ref_img, mod_img):
    # Get DoG keypoints using SIFT
    detector = cv2.SIFT_create()
    ref_kp = detector.detect(ref_img, None)
    mod_kp = detector.detect(mod_img, None)

    mrSize = 6.0
    patch_size = 32
    ref_ps = extract_patches(ref_kp, ref_img, patch_size, mrSize, 'cv2')
    mod_ps = extract_patches(mod_kp, mod_img, patch_size, mrSize, 'cv2')

    # TODO get onnx version of L2Net 
    # Load model 
    onnx_model = onnx.load("onnx_models/L2Net.onnx")
    onnx.checker.check_model(onnx_model)
    onnx_sess = onnxruntime.InferenceSession("onnx_models/L2Net.onnx")
    onnx_input_name = onnx_sess.get_inputs()[0].name

    # Format input and perform inference
    ref_ps = np.array(ref_ps, dtype=np.float32)/255
    ref_ps = ref_ps.reshape(-1, 1, patch_size, patch_size)
    ref_des = onnx_sess.run(None, {onnx_input_name : ref_ps})[0]
    mod_ps = np.array(mod_ps, dtype=np.float32)/255
    mod_ps = mod_ps.reshape(-1, 1, patch_size, patch_size)
    mod_des = onnx_sess.run(None, {onnx_input_name : mod_ps})[0]

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(queryDescriptors=ref_des, 
                            trainDescriptors=mod_des)

    return matches, ref_kp, mod_kp

def hardnet_detect_and_match(ref_img, mod_img):
    # Get DoG keypoints using SIFT
    detector = cv2.SIFT_create()
    ref_kp = detector.detect(ref_img, None)
    mod_kp = detector.detect(mod_img, None)

    mrSize = 6.0
    patch_size = 32
    ref_ps = extract_patches(ref_kp, ref_img, patch_size, mrSize, 'cv2')
    mod_ps = extract_patches(mod_kp, mod_img, patch_size, mrSize, 'cv2')

    # Load model 
    onnx_model = onnx.load(ONNX_PATH + "HardNet++.onnx")
    onnx.checker.check_model(onnx_model)
    onnx_sess = onnxruntime.InferenceSession(ONNX_PATH + "HardNet++.onnx")
    onnx_input_name = onnx_sess.get_inputs()[0].name

    # Format input and perform inference
    ref_ps = np.array(ref_ps, dtype=np.float32)/255
    ref_ps = ref_ps.reshape(-1, 1, patch_size, patch_size)
    ref_des = onnx_sess.run(None, {onnx_input_name : ref_ps})[0]
    mod_ps = np.array(mod_ps, dtype=np.float32)/255
    mod_ps = mod_ps.reshape(-1, 1, patch_size, patch_size)
    mod_des = onnx_sess.run(None, {onnx_input_name : mod_ps})[0]

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(queryDescriptors=ref_des, 
                            trainDescriptors=mod_des)

    return matches, ref_kp, mod_kp

## Match descriptors (bruteforce)
#  Options:
#   - des_type: cv2.NORM_L2 or cv2.NORM_HAMMING for fp and binary descriptors respectively
#   - ratio:    values >= 1.0 match with cross check, values < 1.0 Symmetric matching usign Lowe's ratio test 
#
def match(ref_kp, ref_des, mod_kp, mod_des, des_type=cv2.NORM_L2, ratio=1.0):
    matcher = cv2.BFMatcher(des_type, crossCheck=(ratio >= 1.0))
    good_matches = list()
    if ratio < 1.0:
        matches = matcher.knnMatch(queryDescriptors=ref_des, 
                                   trainDescriptors=mod_des,
                                   k=2)
        for m, n in matches:
            if m.distance < ratio*n.distance:
                good_matches.append(m)

        matches = matcher.knnMatch(queryDescriptors=mod_des, 
                                   trainDescriptors=ref_des,
                                   k=2)
        for m, n in matches:
            if m.distance < ratio*n.distance:
                m.queryIdx, m.trainIdx = m.trainIdx, m.queryIdx
                good_matches.append(m)

    else:
        matches = matcher.match(queryDescriptors=ref_des, 
                                trainDescriptors=mod_des)
        good_matches = matches

    return good_matches

