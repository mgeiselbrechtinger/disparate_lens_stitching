################################################################################
#
#   Applies homographies to images and composes them around base image
#
#
################################################################################

import argparse

import cv2
import numpy as np

OUTPUT_SIZE = [1024, 640]

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('base_img_name', help="Path to base image")
    parser.add_argument('img_hg_pairs', nargs='+', help="Paths to images and homographies pairs")
    parser.add_argument('-t', '--top', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if len(args.img_hg_pairs) % 2 != 0:
        raise AttributeError("Image or homography file missing")

    args.hg_names = args.img_hg_pairs[1::2]
    args.img_names = args.img_hg_pairs[::2]
    # Load base image
    base_img = cv2.imread(args.base_img_name, cv2.IMREAD_UNCHANGED)
    if base_img is None:
        raise OSError(-1, "Could not open file.", args.base_img_name)

    base_rois = np.float32([[0, 0], [base_img.shape[1], base_img.shape[0]]])
    base_aspect = base_img.shape[1]/base_img.shape[0]

    imgs = list()
    hgs = list()
    rois = base_rois.reshape(-1, 1, 2)

    # Load images and calculate ROIs
    for img_name, hg_name in zip(args.img_names, args.hg_names):
        # Load image and Homography
        img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise OSError(-1, "Could not open file.", img_name)
            print("Couldn't load image")
            sys.exit(-1)

        H = np.loadtxt(hg_name, delimiter=',')
        if H.shape != (3, 3):
            raise Exception("Homography of wrong format {H.shape} but expected (3, 3)")

        roi = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]) 
        roi_warped = cv2.perspectiveTransform(roi.reshape(-1, 1, 2), H)

        imgs.append(img)
        hgs.append(H)
        rois = np.concatenate([rois, roi_warped], axis=0)

    # Calculate ROI bound union 
    lower_bound = np.amin(rois, axis=0)[0].astype(np.int32)
    upper_bound = np.amax(rois, axis=0)[0].astype(np.int32) - np.array(base_img.shape)[[1,0]] 
    frame_offset = np.abs(lower_bound).tolist()
    # Translation to frame coordinates as Affine transform
    A = np.float32([[1, 0, frame_offset[0]], [0, 1, frame_offset[1]], [0, 0, 1]])

    # Create common frame from base image
    frame = cv2.copyMakeBorder(base_img, frame_offset[1], upper_bound[1], 
                                         frame_offset[0], upper_bound[0], 
                                         borderType=cv2.BORDER_CONSTANT, value=0)

    # Align images on frame
    for img, H in zip(imgs, hgs):
        # Add alpha channel for overlay region
        alpha = 255*np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
    
        # Apply homography and translation to frame
        H = A.dot(H)
        img = cv2.warpPerspective(img, H, (frame.shape[1], frame.shape[0])) 
        # Overlay region given by warped alpha channel
        mask = (img[:, :, 3] == 255)
        frame[mask, :3] = img[mask, :3]

    # Put base image on top
    if args.top:
        frame[frame_offset[1] : frame_offset[1] + base_img.shape[0], 
              frame_offset[0] : frame_offset[0] + base_img.shape[1]] = base_img

    # Output frame
    frame_size = np.int32(OUTPUT_SIZE)
    if base_aspect > 1: 
        frame_size[1] = int(frame_size[0]/base_aspect)

    else:
        frame_size[0] = int(frame_size[1]*base_aspect)

    frame = cv2.resize(frame, tuple(frame_size))
    cv2.imshow("composition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO dont think is needed
def focal_from_homography(H):
    h = H.flatten()

    f0_a = (h[5]**2 - h[2]**2)/(h[0]**2 + h[1]**2 - h[3]**2 - h[4]**2)
    f0_b = -h[2]*h[5]/(h[0]*h[3] + h[1]*h[4])
    if f0_a > 0 and f0_b > 0:
        f0 = min(f0_a, f0_b)
    elif f0_a > 0:
        f0 = f0_a
    elif f0_b > 0:
        f0 = f0_b
    else:
        raise ValueError("Focal lenght estimation failed")

    f1_a = (h[6]**2 - h[7]**2)/(h[1]**2 + h[4]**2 - h[0]**2 - h[3]**2)
    f1_b = -(h[0]*h[1] + h[3]*h[4])/(h[6]*h[7])
    if f1_a > 0 and f1_b > 0:
        f1 = min(f1_a, f1_b)
    elif f1_a > 0:
        f1 = f1_a
    elif f1_b > 0:
        f1 = f1_b
    else:
        raise ValueError("Focal lenght estimation failed")

    f0 = np.sqrt(f0)
    f1 = np.sqrt(f1)
    print(f1_a, f1_b, f0_a, f0_b)
    f = np.sqrt(f0*f1)
    print(f)
    return f

if __name__=="__main__":
    main()
