################################################################################
#
#   Feature matching library
#
#
################################################################################

import cv2
import math
import numpy as np
from extract_patches.core import extract_patches

ONNX_PATH = "./onnx_models/"
MAX_FEATURES = 4000
NO_DESC  = False

def orb_detect_and_match(ref_img, mod_img):
    nOctaves = round(math.log(min(ref_img.shape), 2) - 2)
    #detector = cv2.ORB_create(nfeatures=MAX_FEATURES, nlevels=7, scaleFactor=1.3)
    detector = cv2.ORB_create(nfeatures=MAX_FEATURES, nlevels=nOctaves)
    #detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
    
    if NO_DESC:
        ref_kp = detector.detect(ref_img, None)
        mod_kp = detector.detect(mod_img, None)

        mrSize = 1.0
        patch_size = 16
        ref_ps = extract_patches(ref_kp, ref_img, patch_size, mrSize, 'cv2')
        ref_ps = np.array(ref_ps, dtype=np.float32)
        ref_des = ref_ps.reshape(-1, patch_size**2)
        ref_des = (ref_des - np.mean(ref_des, axis=1)[:, None])/np.std(ref_des, axis=1)[:, None]

        mod_ps = extract_patches(mod_kp, mod_img, patch_size, mrSize, 'cv2')
        mod_ps = np.array(mod_ps, dtype=np.float32)
        mod_des = mod_ps.reshape(-1, patch_size**2)
        mod_des = (mod_des - np.mean(mod_des, axis=1)[:, None])/np.std(mod_des, axis=1)[:, None]

        matches = match(ref_kp, ref_des, mod_kp, mod_des)
        return matches, ref_kp, mod_kp

    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    matches = match(ref_kp, ref_des, mod_kp, mod_des, des_type=cv2.NORM_HAMMING)
    return matches, ref_kp, mod_kp

def brisk_detect_and_match(ref_img, mod_img):
    nOctaves = round(math.log(min(ref_img.shape), 2) - 2)
    detector = cv2.BRISK_create(octaves=nOctaves)
    #detector = cv2.BRISK_create()

    if NO_DESC:
        ref_kp, _ = filter_keypoints(detector.detect(ref_img, None))
        mod_kp, _ = filter_keypoints(detector.detect(mod_img, None))

        mrSize = 1.0
        patch_size = 16
        ref_ps = extract_patches(ref_kp, ref_img, patch_size, mrSize, 'cv2')
        ref_ps = np.array(ref_ps, dtype=np.float32)
        ref_des = ref_ps.reshape(-1, patch_size**2)
        ref_des = (ref_des - np.mean(ref_des, axis=1)[:, None])/np.std(ref_des, axis=1)[:, None]

        mod_ps = extract_patches(mod_kp, mod_img, patch_size, mrSize, 'cv2')
        mod_ps = np.array(mod_ps, dtype=np.float32)
        mod_des = mod_ps.reshape(-1, patch_size**2)
        mod_des = (mod_des - np.mean(mod_des, axis=1)[:, None])/np.std(mod_des, axis=1)[:, None]

        matches = match(ref_kp, ref_des, mod_kp, mod_des)
        return matches, ref_kp, mod_kp
    
    ref_kp, ref_des = filter_keypoints(*detector.detectAndCompute(ref_img, None))
    mod_kp, mod_des = filter_keypoints(*detector.detectAndCompute(mod_img, None))

    matches = match(ref_kp, ref_des, mod_kp, mod_des, des_type=cv2.NORM_HAMMING)
    return matches, ref_kp, mod_kp

def akaze_detect_and_match(ref_img, mod_img):
    nOctaves = round(math.log(min(ref_img.shape), 2) - 2)
    detector = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE_UPRIGHT, nOctaves=nOctaves, nOctaveLayers=3, threshold=0.00005)
    #detector = cv2.AKAZE_create()

    if NO_DESC:
        ref_kp, _ = filter_keypoints(detector.detect(ref_img, None))
        mod_kp, _ = filter_keypoints(detector.detect(mod_img, None))

        mrSize = 3.0
        patch_size = 16
        ref_ps = extract_patches(ref_kp, ref_img, patch_size, mrSize, 'cv2')
        ref_ps = np.array(ref_ps, dtype=np.float32)
        ref_des = ref_ps.reshape(-1, patch_size**2)
        ref_des = (ref_des - np.mean(ref_des, axis=1)[:, None])/np.std(ref_des, axis=1)[:, None]

        mod_ps = extract_patches(mod_kp, mod_img, patch_size, mrSize, 'cv2')
        mod_ps = np.array(mod_ps, dtype=np.float32)
        mod_des = mod_ps.reshape(-1, patch_size**2)
        mod_des = (mod_des - np.mean(mod_des, axis=1)[:, None])/np.std(mod_des, axis=1)[:, None]

        matches = match(ref_kp, ref_des, mod_kp, mod_des)
        return matches, ref_kp, mod_kp
    
    ref_kp, ref_des = filter_keypoints(*detector.detectAndCompute(ref_img, None))
    mod_kp, mod_des = filter_keypoints(*detector.detectAndCompute(mod_img, None))

    matches = match(ref_kp, ref_des, mod_kp, mod_des)#, des_type=cv2.NORM_HAMMING)
    return matches, ref_kp, mod_kp

def surf_detect_and_match(ref_img, mod_img):
    nOctaves = round(math.log(min(ref_img.shape), 2) - 2)
    #detector = cv2.xfeatures2d.SURF_create(nOctaves=nOctaves, extended=True, upright=True)
    detector = cv2.xfeatures2d.SURF_create()
    
    ref_kp, ref_des = filter_keypoints(*detector.detectAndCompute(ref_img, None))
    mod_kp, mod_des = filter_keypoints(*detector.detectAndCompute(mod_img, None))

    matches = match(ref_kp, ref_des, mod_kp, mod_des)
    return matches, ref_kp, mod_kp

def sift_detect_and_match(ref_img, mod_img):
    #detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    detector = cv2.SIFT_create()
   
    ref_kp, ref_des = detector.detectAndCompute(ref_img, None)
    mod_kp, mod_des = detector.detectAndCompute(mod_img, None)

    #ref_kp, ref_des = filter_keypoints(ref_kp, ref_des)
    ref_kp, ref_des = remove_upsampled_octaves(ref_kp, ref_des)
    mod_kp, mod_des = remove_upsampled_octaves(mod_kp, mod_des)
    #mod_kp, mod_des = filter_keypoints(mod_kp, mod_des)

    # RootSIFT version 
    #ref_des = np.sqrt(ref_des)#/np.linalg.norm(ref_des, ord=1, axis=0))
    #mod_des = np.sqrt(mod_des)#/np.linalg.norm(mod_des, ord=1, axis=0))

    matches = match(ref_kp, ref_des, mod_kp, mod_des)
    return matches, ref_kp, mod_kp

def test_detect_and_match(ref_img, mod_img):
    # reference image is of smaller scale
    # modified image is of higher scale
    # downscale modified image and use single scale kp detector
    #ru = 1.0
    #rd = 3.0 # works for values [2.5, 3.7]

    #h, w = ref_img.shape
    #usize = round(w*ru), round(h*ru)
    #ref_img_r = cv2.resize(ref_img, usize, interpolation=cv2.INTER_CUBIC)

    #h, w = mod_img.shape
    #dsize = round(w/rd), round(h/rd)
    #mod_img_r = cv2.resize(mod_img, dsize, interpolation=cv2.INTER_AREA)

    ## use simple single scale kp detector
    ##detector = cv2.GFTTDetector_create(maxCorners=MAX_FEATURES)
    #detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(numOctaves=1, num_layers=2)
    #ref_kp = detector.detect(ref_img_r, None)
    #mod_kp = detector.detect(mod_img_r, None)

    ## SIFT descriptors
    ##descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    #descriptor = cv2.SIFT_create()
    #ref_kp, ref_des = descriptor.compute(ref_img_r, ref_kp)
    #mod_kp, mod_des = descriptor.compute(mod_img_r, mod_kp)

    ## upsample mod keypoints
    #for i in range(len(mod_kp)):
    #    x, y = mod_kp[i].pt
    #    mod_kp[i].pt = x*rd, y*rd 
    ## downscale ref keypoints
    #for i in range(len(ref_kp)):
    #    x, y = ref_kp[i].pt
    #    ref_kp[i].pt = x/ru, y/ru

    from extract_patches.core import extract_patches

    # Get DoG keypoints using SIFT
    detector = cv2.SIFT_create()
    ref_kp = detector.detect(ref_img, None)
    mod_kp = detector.detect(mod_img, None)
    ref_kp, _ = remove_upsampled_octaves(ref_kp, None)
    mod_kp, _ = remove_upsampled_octaves(mod_kp, None)

    mrSize = 6.0
    patch_size = 16
    ref_ps = extract_patches(ref_kp, ref_img, patch_size, mrSize, 'cv2')
    ref_ps = np.array(ref_ps, dtype=np.float32)
    ref_des = ref_ps.reshape(-1, patch_size**2)
    ref_des = (ref_des - np.mean(ref_des, axis=1)[:, None])/np.std(ref_des, axis=1)[:, None]
    #ref_des -= np.amin(ref_des, axis=1)[:, None]
    #ref_des /= np.amax(ref_des, axis=1)[:, None]

    mod_ps = extract_patches(mod_kp, mod_img, patch_size, mrSize, 'cv2')
    mod_ps = np.array(mod_ps, dtype=np.float32)
    mod_des = mod_ps.reshape(-1, patch_size**2)
    mod_des = (mod_des - np.mean(mod_des, axis=1)[:, None])/np.std(mod_des, axis=1)[:, None]
    #mod_des -= np.amin(mod_des, axis=1)[:, None]
    #mod_des /= np.amax(mod_des, axis=1)[:, None]

    matches = match(ref_kp, ref_des, mod_kp, mod_des)
    return matches, ref_kp, mod_kp

def hardnet_detect_and_match(ref_img, mod_img):
    return onnx_detect_and_match(ref_img, mod_img, "HardNet++")

def sosnet_detect_and_match(ref_img, mod_img):
    return onnx_detect_and_match(ref_img, mod_img, "SOSNet")

def onnx_detect_and_match(ref_img, mod_img, model_name):
    import onnx
    import onnxruntime
    from extract_patches.core import extract_patches

    # Get DoG keypoints using SIFT
    #detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    detector = cv2.SIFT_create()
    ref_kp = detector.detect(ref_img, None)
    mod_kp = detector.detect(mod_img, None)

    ref_kp, _ = remove_upsampled_octaves(ref_kp, None)
    mod_kp, _ = remove_upsampled_octaves(mod_kp, None)

    mrSize = 6.0
    patch_size = 32
    ref_ps = extract_patches(ref_kp, ref_img, patch_size, mrSize, 'cv2')
    mod_ps = extract_patches(mod_kp, mod_img, patch_size, mrSize, 'cv2')

    # Load model 
    onnx_model = onnx.load(ONNX_PATH + model_name + ".onnx")
    onnx.checker.check_model(onnx_model)
    onnx_sess = onnxruntime.InferenceSession(ONNX_PATH + model_name + ".onnx")
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

def filter_keypoints(kp, des=None, k_best=MAX_FEATURES):
    if len(kp) <= k_best:
        return kp, des

    idxs = np.argsort([k.response for k in kp])
    kp = np.array(kp)[idxs][:k_best]

    if not des is None:
        des = np.array(des)[idxs][:k_best]

    return kp, des

def remove_upsampled_octaves(kp, des=None):
    octvs = np.array([k.octave & 255 for k in kp])
    mask = (octvs < 128)
    if not des is None:
        des = np.array(des)[mask]
    return filter_keypoints(np.array(kp)[mask], des)

