################################################################################
#
#   Applies homographies to images and composes them around base image
#
#
################################################################################

import argparse

import cv2
import numpy as np
from math import log

OUTPUT_SIZE = [1024, 640]
DEBUG_ROIS = False
DEBUG_BOUNDS = False

def compose(base_img, imgs, hgs, base_on_top=True):
    # Calculate ROIs
    base_rois = np.float32([[0, 0], [base_img.shape[1], base_img.shape[0]]])
    rois = base_rois.reshape(-1, 1, 2)
    for img, H in zip(imgs, hgs):
        roi = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]) 
        roi_warped = cv2.perspectiveTransform(roi.reshape(-1, 1, 2), H)

        rois = np.concatenate([rois, roi_warped], axis=0)

    # TODO Add bounding rois check bound computation
    if DEBUG_BOUNDS:
        rois = np.float32([[-1000, -1000], [6000, 6000]]).reshape(-1,1,2)

    # Calculate ROI bound union 
    lower_bound = np.amin(rois, axis=0)[0].astype(np.int32)
    upper_bound = np.amax(rois, axis=0)[0].astype(np.int32) 
    lower_offset = np.abs(lower_bound)
    upper_offset = upper_bound - np.array(base_img.shape)[[1,0]] 
    # Translation to frame coordinates as Affine transform
    A = np.float32([[1, 0, lower_offset[0]], [0, 1, lower_offset[1]], [0, 0, 1]])

    # Create common frame from base image
    frame = cv2.copyMakeBorder(base_img, lower_offset[1], upper_offset[1], 
                                         lower_offset[0], upper_offset[0], 
                                         borderType=cv2.BORDER_CONSTANT, value=0)

    # Align images on frame
    for img, H in zip(imgs, hgs):
        # Add alpha channel for overlay region
        alpha = 255*np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img_warp = np.concatenate([img, alpha], axis=2)
    
        # Apply homography and translation to frame
        AH = A.dot(H)
        img_warp = cv2.warpPerspective(img_warp, AH, (frame.shape[1], frame.shape[0])) 
        # Overlay region given by warped alpha channel
        mask = (img_warp[:, :, 3] == 255)
        frame[mask, :3] = img_warp[mask, :3]

    # Put base image on top
    if base_on_top:
        frame[lower_offset[1] : lower_offset[1] + base_img.shape[0], 
              lower_offset[0] : lower_offset[0] + base_img.shape[1]] = base_img

    if DEBUG_ROIS:
        for img, H in zip(imgs, hgs):
            AH = A.dot(H)
            roi = np.float32([[0, 0], [img.shape[1], 0], 
                    [img.shape[1], img.shape[0]], [0, img.shape[0]]])
            roi_warped = cv2.perspectiveTransform(roi.reshape(-1, 1, 2), AH)
            frame = cv2.polylines(frame, [roi_warped.astype(np.int32)], True, [0, 0, 255])

    # Output frame
    frame_size = np.int32(OUTPUT_SIZE)
    base_aspect = base_img.shape[1]/base_img.shape[0]
    if base_aspect > 1: 
        frame_size[1] = int(frame_size[0]/base_aspect)

    else:
        frame_size[0] = int(frame_size[1]*base_aspect)

    #frame = cv2.resize(frame, tuple(frame_size))
    return frame


if __name__=="__main__":
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('base_img_name', help="Path to base image")
    parser.add_argument('img_hg_pairs', nargs='+', help="Paths to images and homographies pairs")
    parser.add_argument('-t', '--top', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-s', '--scale', type=int, choices=[1, 2, 4, 8, 16, 32], default=1)
    args = parser.parse_args()

    if len(args.img_hg_pairs) % 2 != 0:
        raise AttributeError("Image or homography file missing")

    args.hg_names = args.img_hg_pairs[1::2]
    args.img_names = args.img_hg_pairs[::2]
    # Load base image
    base_img = cv2.imread(args.base_img_name, cv2.IMREAD_UNCHANGED)
    if base_img is None:
        raise OSError(-1, "Could not open file.", args.base_img_name)

    for i in range(int(log(args.scale, 2))):
        base_img = cv2.pyrDown(base_img)

    # Load images and homographies
    imgs = list()
    hgs = list()
    for img_name, hg_name in zip(args.img_names, args.hg_names):
        # Load image and Homography
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise OSError(-1, "Could not open file.", img_name)
            print("Couldn't load image")
            sys.exit(-1)

        for i in range(int(log(args.scale, 2))):
            img = cv2.pyrDown(img)
            
        H = np.loadtxt(hg_name, delimiter=',')
        if H.shape != (3, 3):
            raise Exception("Homography of wrong format {H.shape} but expected (3, 3)")

        H[0, 2] /= args.scale 
        H[1, 2] /= args.scale 
        H[2, 0] *= args.scale 
        H[2, 1] *= args.scale 

        imgs.append(img)
        hgs.append(H)
        
    mosaic = compose(base_img, imgs, hgs, base_on_top=args.top)

    cv2.namedWindow("composition", cv2.WINDOW_NORMAL)        
    cv2.imshow("composition", mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
