from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    current_path = Path(__file__).parent
    img_num = 0
    img_name = f"{current_path}/data/pattern/image{img_num:03d}.jpg"
    ref_img = cv2.imread(img_name)
    if ref_img is None:
        raise FileNotFoundError("Couldn't load image: " + img_name)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)  

    # TODO modify image
    mod_img = ref_img.copy()

    # Get ORB keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=500)
    ref_kp, ref_des = orb.detectAndCompute(ref_img, None)
    mod_kp, mod_des = orb.detectAndCompute(mod_img, None)

    #kp_img = cv2.drawKeypoints(ref_img, ref_kp, None)
    #plt.figure()
    #plt.imshow(kp_img)
    #plt.show()

    # TODO check params/ use bruteforce matching
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
