import sys
from pathlib import Path

import cv2
import numpy as np

REF = "orb" # "homography" | "sift" | "orb" 
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
    ref_path = '/'.join(img1_name.split('/')[:-1]) + f"/{REF}{img1_num:03d}_{img2_num:03d}.csv"

    # Load files
    img1 = cv2.imread(img1_name, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_name, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print("Couldn't load images")
        sys.exit(-1)

    # img2 = H(img1)
    H = np.loadtxt(ref_path, delimiter=',')
    print(H)

    # Compose images
    frame = cv2.copyMakeBorder(img2, img2.shape[0]//4, img2.shape[0]//4, img2.shape[1]//2, img2.shape[1]//2, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img1 = cv2.copyMakeBorder(img1, img1.shape[0]//4, img1.shape[0]//4, img1.shape[1]//2, img1.shape[1]//2, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img1_warp = cv2.warpPerspective(img1, H, (frame.shape[1], frame.shape[0])) #, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    frame |= img1_warp
    cv2.imshow("composition", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO transform corners to get resulting image size
    #img1_corners = np.array([[0, 0], [img1.shape[1]-1, 0], [0, img1.shape[0]-1], [img1.shape[1]-1, img1.shape[0]-1]], dtype=np.float32)
    #img1_warp_corners = cv2.perspectiveTransform(img1_corners, np.array(H_r, dtype=np.float32))
    #img1_warp_size = (int(np.amax(img1_warp_corners[:,0])), int(np.amax(img1_warp_corners[:,1])))

if __name__=="__main__":
    main()
