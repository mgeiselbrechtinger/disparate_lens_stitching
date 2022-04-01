################################################################################
#
#   Applies homographies to images and composes them around base image
#
#
################################################################################

import argparse

import cv2
import numpy as np

def main():
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('base_img_name', help="Paths to base image")
    parser.add_argument('img_names', nargs='+', help="Paths to stitch images")
    parser.add_argument('hg_names', nargs='+', help="Paths to homography csv-files")
    parser.add_argument('-t', '--top', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if len(args.img_names) != len(args.hg_names):
        raise AttributeError("Image or homography file missing")

    # Load bas image
    base_img = cv2.imread(args.base_img_name, cv2.IMREAD_UNCHANGED)
    if base_img is None:
        raise OSError(-1, "Could not open file.", args.base_img_name)

    # Create common frame from base image
    frame_offset_x = base_img.shape[1]//2
    frame_offset_y = base_img.shape[0]//4 
    frame = cv2.copyMakeBorder(base_img, frame_offset_y, frame_offset_y, 
                                         frame_offset_x, frame_offset_x, 
                                         borderType=cv2.BORDER_CONSTANT, value=0)

    # Align remaining images to frame
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

        # Add alpha channel with all ones and enlarge image to frame size
        alpha = 255*np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
        img = cv2.copyMakeBorder(img, frame_offset_y, frame_offset_y, 
                                      frame_offset_x, frame_offset_x, 
                                      borderType=cv2.BORDER_CONSTANT, value=0)
    
        # Apply homography and overlay on frame
        img = cv2.warpPerspective(img, H, (frame.shape[1], frame.shape[0])) 
        mask = (img[:, :, 3] == 255)
        frame[mask, :3] = img[mask, :3]

    if args.top:
        alpha = 255*np.ones((base_img.shape[0], base_img.shape[1], 1), dtype=base_img.dtype)
        img = np.concatenate([base_img, alpha], axis=2)
        img = cv2.copyMakeBorder(img, frame_offset_y, frame_offset_y, 
                                      frame_offset_x, frame_offset_x, 
                                      borderType=cv2.BORDER_CONSTANT, value=0)

        mask = (img[:, :, 3] == 255)
        frame[mask, :3] = img[mask, :3]

    cv2.imshow("composition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
