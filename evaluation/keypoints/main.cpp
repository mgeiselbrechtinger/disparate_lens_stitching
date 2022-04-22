#include<cstdlib>
#include<cstring>
#include<vector>
#include<fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

static bool VERBOSE = true;
static bool DEBUG = false;

using namespace cv;
using namespace cv::xfeatures2d;

static int loadHomography(const char *fname, Mat &H);

int main(int argc, char** argv)
{
#ifndef HAVE_OPENCV_XFEATURES2D
    std::cerr << "OpenCV xfeatrues2d module missing\n";
    return -1;
#endif

    if( argc != 3){
        std::cerr << "Usage: " << argv[0] << " <src image> <homography>\n";
        return -1;
    }

    Mat img_src;
    img_src = imread(argv[1], IMREAD_GRAYSCALE);

    if(!img_src.data){
        std::cerr << "No image data found.\n";
        return -1;
    }

    // Load homography
    Mat H;
    if(loadHomography(argv[2], H) != 0){
        std::cerr << "Could not open homography file\n";
        return -1;
    }

    Mat img_dest;
    Size dsize = Size(1*img_src.cols, 1*img_src.rows);
    warpPerspective(img_src, img_dest, H, dsize);

    if(DEBUG){
        namedWindow("Display Warp", WINDOW_NORMAL);
        imshow("Display Warp", img_dest);
        waitKey(0);
    }

    // TODO add more and make dynamic selection
    //Ptr<ORB> detector = ORB::create(2500);
    //Ptr<AKAZE> detector = AKAZE::create();
    //Ptr<BRISK> detector = BRISK::create();
    Ptr<SIFT> detector = SIFT::create();
    //Ptr<SURF> detector = SURF::create(400);
    //Ptr<HarrisLaplaceFeatureDetector> detector = HarrisLaplaceFeatureDetector::create();
    
    // Wrapper for affinity invariance
    //Ptr<AffineFeature> detector = AffineFeature::create(SIFT::create(200));
    
    std::vector<KeyPoint> kp_src, kp_dest;
    detector->detect(img_src, kp_src);
    detector->detect(img_dest, kp_dest);

    if(VERBOSE | DEBUG){
        Mat img_src_kpts, img_dest_kpts;
        drawKeypoints(img_src, kp_src, img_src_kpts, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(img_dest, kp_dest, img_dest_kpts, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        Mat img_kpts;
        //copyMakeBorder(img_src_kpts, img_src_kpts, 0, img_src.rows, 0, img_src.cols, BORDER_CONSTANT, Scalar(0));
        hconcat(img_src_kpts, img_dest_kpts, img_kpts);
        namedWindow("Display Keypoints", WINDOW_NORMAL);
        imshow("Display Keypoints", img_kpts);
        waitKey(0);
    }

    // Count keypoints in overlapping region
    //Mat mask = Mat::ones(img_src.size(), CV_8U);
    //Mat mask_warp = Mat::ones(img_dest.size(), CV_8U); 
    //warpPerspective(mask_warp, mask_warp, H, mask_warp.size());
    //bitwise_and(mask, mask_warp, mask);
    //int inlier_cnt = 0;
    //int outlier_cnt = 0;
    //for(size_t i = 0; i < kp_src.size(); i++){
    //    Point2i pt = kp_src[i].pt;
    //    if(mask.at<uint8_t>(pt.y, pt.x) == 1)
    //        inlier_cnt++;
    //    else
    //        outlier_cnt++;
    //}
    //std::cout << "outliers: " << outlier_cnt;
    //std::cout << "\tinliers: " << inlier_cnt << std::endl;

    //if(DEBUG){
    //    namedWindow("Overlap region", WINDOW_NORMAL);
    //    imshow("Overlap region", 255*mask);
    //    waitKey(0);
    //}

    // Test evalutation
    float repeatability = 0;
    int correspondences = 0; 
    evaluateFeatureDetector(img_src, img_dest, H, &kp_src, &kp_dest, repeatability, correspondences, detector);

    // TODO refine output
    std::cout << "repeatability,correpondences,source_keypoints,destination_keypoints\n";
    std::cout << repeatability << "," << correspondences << "," << kp_src.size() << "," << kp_dest.size() << std::endl;
    
    return 0;
}

static int loadHomography(const char *fname, Mat &H)
{
    std::ifstream hfile{fname};
    if(!hfile) {
        std::cerr << "Could not open homography file\n";
        return -1;
    }

    std::vector<float> data;
    char line[256];
    char *elem;
    while(hfile.getline(line, 256, '\n')){
        elem = std::strtok(line, ",");
        while(elem != NULL){
            data.push_back(std::strtof(elem, NULL));
            elem = std::strtok(NULL, ",");
        }
    }

    H = Mat(data, true).reshape(1, std::vector<int>(2, 3));
    if(DEBUG)
        std::cout << "H: " << H << std::endl;
    
    return 0;
}
