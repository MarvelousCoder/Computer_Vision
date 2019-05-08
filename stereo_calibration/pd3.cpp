#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

vector<pair<int, int>> cliques;

void on_mouse(int event, int x, int y, int flags, void* userdata) {
    if ( event == EVENT_LBUTTONDOWN )
        cliques.push_back(make_pair(x, y));
}

void estima_medidas_warrior(Mat Q)
{
    Mat img = imread("teste2.png", -1);
    namedWindow("Req3", WINDOW_NORMAL);
    resizeWindow("Req3", 1000,  1000);
    while(1) {
        setMouseCallback("Req3", on_mouse);
        imshow( "Req3", img);
        if((char)waitKey(55) == 27) break;
    }
    Mat out;
    reprojectImageTo3D(img, out, Q);
   
    for(int i = 0; i < cliques.size(); i+=2)
    {
        Vec3b pt1 = out.at<Vec3b>(cliques[i].second, cliques[i].first);
        Vec3b pt2 = out.at<Vec3b>(cliques[i+1].second, cliques[i+1].first);
        if(i == 0) cout << "largura em mm: " << norm(pt1, pt2, NORM_L2) << endl;
        else if(i == 2) cout << "altura em mm: " << norm(pt1, pt2, NORM_L2) << endl;
        else if(i == 4) cout << "profundidade em mm: " << norm(pt1, pt2, NORM_L2) << endl;
    }
}

int main () {
    Mat R_r(3,3, CV_64FC1), R_l(3,3, CV_64FC1);
    Mat T_r(3,1, CV_64FC1), T_l(3,1, CV_64FC1);
    Mat R_te(3, 1, CV_64FC1);

    R_te.at<double>(0, 0) = 1.918835;
    R_te.at<double>(1, 0) = 0.768994;
    R_te.at<double>(2, 0) = -0.527060;    
    
    Rodrigues(R_te, R_l);
    R_te.at<double>(0, 0) = 1.797318;
    R_te.at<double>(1, 0) = 1.018771;
    R_te.at<double>(2, 0) = -0.713289; 
    Rodrigues(R_te, R_r);

    T_r.at<double>(0, 0) = -614.549000;
    T_r.at<double>(1, 0) = 193.240700;
    T_r.at<double>(2, 0) = 3242.754000;

    T_l.at<double>(0, 0) = -532.285900;
    T_l.at<double>(1, 0) = 207.183600;
    T_l.at<double>(2, 0) = 2977.408000;

    Mat cam_matrix_l(3, 3, CV_64FC1), cam_matrix_r(3, 3, CV_64FC1);

    cam_matrix_l.at<double>(0, 0) = 6704.926882; // frectl
    cam_matrix_l.at<double>(0, 1) = 0.000103; // alphal
    cam_matrix_l.at<double>(0, 2) = 838.251932; // cxl
    cam_matrix_l.at<double>(1, 0) = 0.0; 
    cam_matrix_l.at<double>(1, 1) = 6705.241311; // fyl
    cam_matrix_l.at<double>(1, 2) = 857.560286; // cyl 
    cam_matrix_l.at<double>(2, 0) = 0.0;
    cam_matrix_l.at<double>(2, 1) =  0.0;
    cam_matrix_l.at<double>(2, 2) = 1.0;
    
    cam_matrix_r.at<double>(0, 0) = 6682.125964; // frectr
    cam_matrix_r.at<double>(0, 1) = 0.000101; // alphar
    cam_matrix_r.at<double>(0, 2) = 1075.207200; // cxr
    cam_matrix_r.at<double>(1, 0) = 0.0; 
    cam_matrix_r.at<double>(1, 1) = 6681.475962; // fyr
    cam_matrix_r.at<double>(1, 2) = 757.700292; // cyr
    cam_matrix_r.at<double>(2, 0) = 0.0;
    cam_matrix_r.at<double>(2, 1) =  0.0;
    cam_matrix_r.at<double>(2, 2) = 1.0;

    Mat R = R_r * R_l.t();
    Mat T = T_r - R * T_l;

    Mat warrior_l = imread("../data/warriorL.jpg", 0);
    Mat warrior_r = imread("../data/warriorR.jpg", 0);
    Size image_size_l = warrior_l.size(); 
    Mat R1, R2, P1, P2, Q;
    stereoRectify(cam_matrix_l, noArray(), cam_matrix_r, 
                    noArray(), image_size_l, R, T, R1, R2, P1, P2, Q, 0);
    bool isVerticalStereo = false;
    isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    Mat l1, l2, r1, r2, imgl, imgr;
    initUndistortRectifyMap(cam_matrix_l, noArray(), R1, P1, image_size_l, CV_16SC2, l1, l2);
    initUndistortRectifyMap(cam_matrix_r, noArray(), R2, P2, image_size_l, CV_16SC2, r1, r2);
    Mat img_r, img_l;
    remap(warrior_l, img_l, l1, l2, INTER_LINEAR);
    remap(warrior_r, img_r, r1, r2, INTER_LINEAR);

    cv::Mat pair;
    if (!isVerticalStereo)
      pair.create(image_size_l.height, image_size_l.width * 2, CV_8UC3);
    else
      pair.create(image_size_l.height * 2, image_size_l.width, CV_8UC3);
    if (!isVerticalStereo) {
        cv::Mat part = pair.colRange(0, image_size_l.width);
        cvtColor(img_l, part, cv::COLOR_GRAY2BGR);
        part = pair.colRange(image_size_l.width, image_size_l.width * 2);
        cvtColor(img_r, part, cv::COLOR_GRAY2BGR);
        for (int j = 0; j < image_size_l.height; j += 16)
            cv::line(pair, cv::Point(0, j), cv::Point(image_size_l.width * 2, j),
                    cv::Scalar(0, 255, 0));
    } else {
        cv::Mat part = pair.rowRange(0, image_size_l.height);
        cv::cvtColor(img_l, part, cv::COLOR_GRAY2BGR);
        part = pair.rowRange(image_size_l.height, image_size_l.height * 2);
        cv::cvtColor(img_r, part, cv::COLOR_GRAY2BGR);
        for (int j = 0; j < image_size_l.width; j += 16)
            line(pair, cv::Point(j, 0), cv::Point(j, image_size_l.height * 2),
                cv::Scalar(0, 255, 0));
    }
    imshow("rectified", pair);
    imwrite("pair_warrior.jpg", pair);
    
    imshow("imgl", img_l);
    imwrite("imgl.jpg", img_l);
    imshow("imgr", img_r);
    imwrite("imgr.jpg", img_r);
    while((char)waitKey(0) != 27);

    int max_disp = 160;
    double lambda = 8000.0;
    double sigma  = 1.5;
    double vis_mult = 1.0;
    int wsize = 7; 
    Mat left, right;
    img_l.convertTo(left, CV_8U);
    img_r.convertTo(right, CV_8U);
    
    Ptr<DisparityWLSFilter> wls_filter;
    Mat left_for_matcher, right_for_matcher;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    right_for_matcher = left.clone();
    left_for_matcher = right.clone();
    Mat conf_map = Mat(left.rows, left.cols, CV_8U);
    conf_map = Scalar(255);
    double matching_time, filtering_time;

    Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
    left_matcher->setP1(24*wsize*wsize);
    left_matcher->setP2(96*wsize*wsize);
    left_matcher->setPreFilterCap(63);
    left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
    wls_filter = createDisparityWLSFilter(left_matcher);
    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

    matching_time = (double)getTickCount();
    left_matcher->compute(left_for_matcher, right_for_matcher,left_disp);
    right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
    matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();

    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    filtering_time = (double)getTickCount();
    wls_filter->filter(left_disp,left,filtered_disp,right_disp);
    filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    //! [filtering]
    conf_map = wls_filter->getConfidenceMap();

    namedWindow("left", WINDOW_AUTOSIZE);
    imshow("left", left);
    namedWindow("right", WINDOW_AUTOSIZE);
    imshow("right", right);

    //! [visualization]
    Mat raw_disp_vis;
    getDisparityVis(left_disp,raw_disp_vis,vis_mult);
    namedWindow("raw disparity", WINDOW_AUTOSIZE);
    imshow("raw disparity", raw_disp_vis);
    Mat disparity = left_disp.clone();
    double mini, maxi;
    minMaxIdx(left_disp, &mini, &maxi);
    disparity = (disparity - mini) * (255/(maxi - mini));
    disparity.convertTo(disparity, CV_8U);
    imshow("Mapa de disparidade", disparity);
    imwrite("warrior_disp.png", disparity);
    waitKey(0);

    Mat filtered_disp_vis;
    getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
    namedWindow("filtered disparity", WINDOW_AUTOSIZE);
    imshow("filtered disparity", filtered_disp_vis);
    imwrite("warrior_depth.png", filtered_disp_vis);
    waitKey(0);

    // estima_medidas_warrior(Q);
    return 0;
}