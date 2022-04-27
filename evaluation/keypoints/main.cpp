#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

#include "argparse.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

static int loadHomography(const std::string &fname, Mat &H);

int main(int argc, char** argv)
{
#ifndef HAVE_OPENCV_XFEATURES2D
    std::cerr << "OpenCV xfeatrues2d module missing\n";
    return -1;
#endif

    // Parse arguments
    argparse::ArgumentParser parser("Keypoint-eval");

    parser.add_argument("image")
        .help("Path to image file");

    parser.add_argument("homography")
        .help("Path to homography file");

    parser.add_argument("-d", "--detector")
        .help("Select detector")
        .default_value(std::string("sift"));

    parser.add_argument("-i", "--init")
        .help("Initial transform of image");

    parser.add_argument("-v", "--verbose")
        .help("Set output level verbose")
        .default_value(false)
        .implicit_value(true);

    try{
      parser.parse_args(argc, argv);
    }
    catch(const std::runtime_error& err){
      std::cerr << err.what() << std::endl;
      std::cerr << parser;
      std::exit(-1);
    }

    bool verbose = parser.get<bool>("--verbose");

    Mat img_src;
    img_src = imread(parser.get<std::string>("image"), IMREAD_GRAYSCALE);

    if(!img_src.data){
        std::cerr << "No image data found.\n";
        return -1;
    }

    Mat H_src = Mat::eye(Size(3, 3), CV_32F);
    if(auto hname = parser.present("--init")){
        if(loadHomography(*hname, H_src) != 0){
            std::cerr << "Could not open homography file\n";
            return -1;
        }
        warpPerspective(img_src, img_src, H_src, img_src.size());
    }

    Mat H;
    // Load homography
    if(loadHomography(parser.get<std::string>("homography"), H) != 0){
        std::cerr << "Could not open homography file\n";
        return -1;
    }

    H *= H_src.inv();

    Mat img_dest;
    // TODO dobule in size to accomodate for warp stretching
    Size dsize = Size(1*img_src.cols, 1*img_src.rows);
    float dx = 0; //3*img_src.cols/2;
    float dy = 0; //3*img_src.rows;
    Mat A = (Mat_<float>(3, 3) << 1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0);
    H = A*H;
    warpPerspective(img_src, img_dest, H, dsize);

    // Select descriptor
    Ptr<Feature2D> detector;
    Ptr<Feature2D> detector1;
    const std::string &dname = parser.get<std::string>("--detector");
    if(dname == "sift"){
        //detector = SIFT::create(3000, 3, 0.035, 25, 1.6);
        detector = SIFT::create();

    }else if(dname == "orb"){
        detector = ORB::create(500);
  
    }else if(dname == "brisk"){
        detector = BRISK::create(20, 6, 1.1);
   
    }else if(dname == "akaze"){
        detector = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001, 4, 4, KAZE::DIFF_PM_G2);
   
    }else if(dname == "surf"){
        detector = SURF::create();
    
    }else if(dname == "harris-laplace"){
        detector = HarrisLaplaceFeatureDetector::create(6, 0.01, 0.01, 5000, 4);

    }else if(dname == "asift"){
        detector = AffineFeature::create(SIFT::create(200));

    }else{
      std::cerr << "Selected invalid detector\n";
      std::exit(-1);

    }
    
    detector1 = detector;
    std::vector<KeyPoint> kp_src, kp_dest;
    detector->detect(img_src, kp_src);
    detector1->detect(img_dest, kp_dest);

    if(verbose){
        Mat img_src_kpts, img_dest_kpts;
        drawKeypoints(img_src, kp_src, img_src_kpts, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(img_dest, kp_dest, img_dest_kpts, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        Mat img_kpts;
        //copyMakeBorder(img_src_kpts, img_src_kpts, 0, 3*img_src.rows, 0, img_src.cols, BORDER_CONSTANT, Scalar(0));
        //copyMakeBorder(img_dest_kpts, img_dest_kpts, 0, img_dest.rows, 0, img_dest.cols, BORDER_CONSTANT, Scalar(0));
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

    //if(verbose){
    //    namedWindow("Overlap region", WINDOW_NORMAL);
    //    imshow("Overlap region", 255*mask);
    //    waitKey(0);
    //}

    // Test evalutation
    float repeatability = 0;
    int correspondences = 0; 
    evaluateFeatureDetector(img_src, img_dest, H, &kp_src, &kp_dest, repeatability, correspondences, detector);

    repeatability = std::max((double)repeatability, 0.0);
    correspondences = std::max(0, correspondences);

    // TODO refine output
    //std::cout << "repeatability,correspondences,source_keypoints,destination_keypoints\n";
    std::cout << repeatability << "," << correspondences << "," << kp_src.size() << "," << kp_dest.size() << std::endl;
    
    return 0;
}

static int loadHomography(const std::string &fname, Mat &H)
{
    std::ifstream hfile{fname};
    if(!hfile) {
        std::cerr << "Could not open homography file\n";
        return -1;
    }

    std::vector<float> data;

    std::string line;
    while(getline(hfile, line)){
        std::istringstream sline(line);
        std::string token;
        while(getline(sline, token, ',')){
            data.push_back(std::stof(token));
        }
    }

    if(data.size() != 9){
        std::cerr << "Error while reading homography\n";
        return -1;
    }

    H = Mat(data, true).reshape(1, 3);
    
    return 0;
}
