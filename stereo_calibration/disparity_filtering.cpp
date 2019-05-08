#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

struct parameters {
    String left_im;
    String right_im;
    String dst_conf_path;
    String algo;
    String al;
    bool no_display;
    String req_2;
    String req_1;
    int max_disp;
    double lambda;
    double sigma;
    double vis_mult;
    int wsize;
    vector<pair<int, int>> cliques;
};

void calcula_morpheus(parameters params);
void inicia_matrizes_morpheus(Mat* R_l, Mat* R_r, Mat* T_l, Mat* T_r, Mat* cam_matrix_l, Mat* cam_matrix_r);
void cria_mapas_morpheus(Mat left, Mat right, parameters params);
void calcula_mapas(Mat l, Mat r, parameters prms);
void estima_medidas_morpheus(Mat Q, parameters prms);
void on_mouse(int event, int x, int y, int flags, void* userdata);

const String keys =
    "{help h usage ? |                  | print this message                                                }"
    "{@left_a        |                  | left view of the stereopair                                       }"
    "{@right_a       |                  | left view of the stereopair                                       }"
    "{@req_1         |                  | execute or not the requirement one                                }"
    "{@req_2         |                  | execute or not the requirement two                                }"
    "{@algorithm     |                  | stereo matching method (bm or sgbm)                               }"
    "{@al            |                  | aloe or baby?                                                     }"
    "{@window_size   |                  | window size                                                       }"
    "{vis_mult       |1.0               | coefficient used to scale disparity map visualizations            }"
    "{max_disparity  |160               | parameter of stereo matching                                      }"
    "{wls_lambda     |8000.0            | parameter of post-filtering                                       }"
    "{wls_sigma      |1.5               | parameter of post-filtering                                       }"
    ;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc,argv,keys);
    parser.about("Disparity Filtering Demo");
    if (parser.has("help")) 
        return parser.printMessage(), 0;

    parameters params;

    params.left_im = parser.get<String>(0);
    params.right_im = parser.get<String>(1);
    params.req_1 = parser.get<String>(2);
    params.req_2 = parser.get<String>(3);
    params.algo = parser.get<String>(4);
    params.al = parser.get<String>(5);
    params.wsize = parser.get<int>(6);

    params.max_disp = parser.get<int>("max_disparity");
    params.lambda = parser.get<double>("wls_lambda");
    params.sigma  = parser.get<double>("wls_sigma");
    params.vis_mult = parser.get<double>("vis_mult");

    if (!parser.check()) 
        return parser.printErrors(), -1;
    
    if(params.max_disp<=0 || params.max_disp%16!=0)
        return printf("Incorrect max_disparity value: it should be positive and divisible by 16"), -1;

    if(params.wsize<=0 || params.wsize%2!=1) 
        return printf("Incorrect window_size value: it should be positive and odd"), -1;

    //! [load_views]
    Mat left  = imread(params.left_im, IMREAD_COLOR);
    if ( left.empty() )
        return cout << "Cannot read image file: " << params.left_im, -1;

    Mat right = imread(params.right_im, IMREAD_COLOR);
    if ( right.empty() ) 
        return cout << "Cannot read image file: " << params.right_im, -1;
    if(params.algo != "sgbm" and params.algo != "bm") return printf("Unsupported algorithm"), -1;

    if(params.req_1 == "s") calcula_mapas(left, right, params);
    if(params.req_2 == "s") calcula_morpheus(params);
    return 0;
    
}

void on_mouse(int event, int x, int y, int flags, void* userdata) {
    if ( event == EVENT_LBUTTONDOWN ) {
        parameters* params = (parameters*) userdata;
        params->cliques.push_back(make_pair(x, y));
    }
}

void calcula_mapas(Mat l, Mat r, parameters params) 
{
     //! [load_views]
    Mat left_for_matcher, right_for_matcher;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    Mat conf_map = Mat(l.rows, l.cols, CV_8U);
    conf_map = Scalar(255);
    Ptr<DisparityWLSFilter> wls_filter;
    double matching_time, filtering_time;

    
    left_for_matcher  = l.clone();
    right_for_matcher = r.clone();

    if(params.algo=="bm")
    {
        //! [matching]
        Ptr<StereoBM> left_matcher = StereoBM::create(params.max_disp,params.wsize);
        wls_filter = createDisparityWLSFilter(left_matcher);
        Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

        cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
        cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);

        matching_time = (double)getTickCount();
        left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
        right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
        matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
        //! [matching]
    }
    else if(params.algo=="sgbm")
    {
        Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,params.max_disp,params.wsize);
        left_matcher->setP1(24*params.wsize*params.wsize);
        left_matcher->setP2(96*params.wsize*params.wsize);
        left_matcher->setPreFilterCap(63);
        left_matcher->setMode(StereoSGBM::MODE_SGBM);
        wls_filter = createDisparityWLSFilter(left_matcher);
        Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

        matching_time = (double)getTickCount();
        left_matcher-> compute(left_for_matcher, right_for_matcher,left_disp);
        right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
        matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
    }

    //! [filtering]
    wls_filter->setLambda(params.lambda);
    wls_filter->setSigmaColor(params.sigma);
    filtering_time = (double)getTickCount();
    wls_filter->filter(left_disp,l,filtered_disp,right_disp);
    filtering_time = ((double)getTickCount() - filtering_time)/getTickFrequency();
    //! [filtering]
    conf_map = wls_filter->getConfidenceMap();

    //collect and print all the stats:
    cout.precision(2);
    cout << "Matching time:  "<< matching_time << "s" << endl;
    cout << "Filtering time: "<< filtering_time << "s" << endl;

    double MSE_before,percent_bad_before,MSE_after,percent_bad_after;

    namedWindow("left", WINDOW_AUTOSIZE);
    imshow("left", l);
    namedWindow("right", WINDOW_AUTOSIZE);
    imshow("right", r);

    //! [visualization]
    Mat raw_disp_vis;
    getDisparityVis(left_disp, raw_disp_vis, params.vis_mult);
    namedWindow("raw disparity", WINDOW_AUTOSIZE);
    imshow("raw disparity", raw_disp_vis);
    Mat disparity = left_disp.clone();
    double mini, maxi;
    minMaxIdx(left_disp, &mini, &maxi);
    disparity = (disparity - mini) * (255/(maxi - mini));
    disparity.convertTo(disparity, CV_8U);
    imshow("Mapa de disparidade", disparity);
    String s;
    params.al == "a" ? s = "aloe_" + params.algo + "_disp.png" : s = "baby_" + params.algo + "_disp.png";
    imwrite(s, disparity);
    
    //! [visualization]
    Mat depth(filtered_disp.rows, filtered_disp.cols, CV_16SC1);
    for(int j=0; j < filtered_disp.cols; j++) 
    {
        for (int i=0; i < filtered_disp.rows; i++)
        {
            short intensity = filtered_disp.at<short>(Point(j, i));
            if(intensity != 0) depth.at<short>(Point(j, i)) = 3740*160/intensity; 
        }
    }
    
    minMaxIdx(filtered_disp, &mini, &maxi);
    depth = (depth - mini) * (255/(maxi - mini));
    depth.convertTo(depth, CV_8U);
    
    namedWindow("Mapa de Profundidade", WINDOW_NORMAL);
    imshow("Mapa de Profundidade", depth);
    params.al == "a" ? s = "aloe_" + params.algo + "_depth.png" : s = "baby_" + params.algo + "_depth.png";
    imwrite(s, depth);
    waitKey(0);
}

void inicia_matrizes_morpheus(Mat* R_l, Mat* R_r, Mat* T_l, Mat* T_r, Mat* cam_matrix_l, Mat* cam_matrix_r) 
{
    R_r->at<double>(0, 0) = 0.48946344;
    R_r->at<double>(0, 1) = 0.87099159;
    R_r->at<double>(0, 2) = -0.04241701;
    R_r->at<double>(1, 0) = 0.33782142;
    R_r->at<double>(1, 1) = -0.23423702;
    R_r->at<double>(1, 2) = -0.91159734;
    R_r->at<double>(2, 0) = -0.80392924;
    R_r->at<double>(2, 1) =  0.43186419;
    R_r->at<double>(2, 2) = -0.40889007;

    T_r->at<double>(0, 0) = -614.549000;
    T_r->at<double>(1, 0) = 193.240700;
    T_r->at<double>(2, 0) = 3242.754000;

    R_l->at<double>(0, 0) = 0.70717199;
    R_l->at<double>(0, 1) = 0.70613396;
    R_l->at<double>(0, 2) = -0.03581348;
    R_l->at<double>(1, 0) = 0.28815232;
    R_l->at<double>(1, 1) = -0.33409066;
    R_l->at<double>(1, 2) = -0.89741388;
    R_l->at<double>(2, 0) = -0.64565936;
    R_l->at<double>(2, 1) =  0.62430623;
    R_l->at<double>(2, 2) = -0.43973369;

    T_l->at<double>(0, 0) = -532.285900;
    T_l->at<double>(1, 0) = 207.183600;
    T_l->at<double>(2, 0) = 2977.408000;

    cam_matrix_l->at<double>(0, 0) = 6704.926882; // frectl
    cam_matrix_l->at<double>(0, 1) = 0.000103; // alphal
    cam_matrix_l->at<double>(0, 2) = 738.251932; // cxl
    cam_matrix_l->at<double>(1, 0) = 0.0; 
    cam_matrix_l->at<double>(1, 1) = 6705.241311; // fyl
    cam_matrix_l->at<double>(1, 2) = 457.560286; // cyl 
    cam_matrix_l->at<double>(2, 0) = 0.0;
    cam_matrix_l->at<double>(2, 1) =  0.0;
    cam_matrix_l->at<double>(2, 2) = 1.0;
    
    cam_matrix_r->at<double>(0, 0) = 6682.125964; // frectr
    cam_matrix_r->at<double>(0, 1) = 0.000101; // alphar
    cam_matrix_r->at<double>(0, 2) = 875.207200; // cxr
    cam_matrix_r->at<double>(1, 0) = 0.0; 
    cam_matrix_r->at<double>(1, 1) = 6681.475962; // fyr
    cam_matrix_r->at<double>(1, 2) = 357.700292; // cyr
    cam_matrix_r->at<double>(2, 0) = 0.0;
    cam_matrix_r->at<double>(2, 1) =  0.0;
    cam_matrix_r->at<double>(2, 2) = 1.0;
}

void calcula_morpheus(parameters params) 
{
    Mat R_r(3,3, CV_64FC1), R_l(3,3, CV_64FC1);
    Mat T_r(3,1, CV_64FC1), T_l(3,1, CV_64FC1);
    Mat cam_matrix_l(3, 3, CV_64FC1), cam_matrix_r(3, 3, CV_64FC1);
    // Mat K_l(1,5, CV_64FC1), K_r(1,5, CV_64FC1);
    
    // K_l.at<double>(0, 0) = -0.125368;
    // K_l.at<double>(0, 1) = -0.097388;
    // K_l.at<double>(0, 2) = -0.003711;
    // K_l.at<double>(0, 3) = -0.000161;
    // K_l.at<double>(0, 4) = 0.000000;
    // K_r.at<double>(0, 0) = -0.106090;
    // K_r.at<double>(0, 1) = -0.533543;
    // K_r.at<double>(0, 2) = -0.005174;
    // K_r.at<double>(0, 3) =  0.000517;
    // K_r.at<double>(0, 4) = 0.00;

    inicia_matrizes_morpheus(&R_l, &R_r, &T_l, &T_r, &cam_matrix_l, &cam_matrix_r);

    Mat R = R_r * R_l.t();
    Mat T = T_r - R * T_l;

    Mat morpheus_l = imread("data/MorpheusL.jpg", 0);
    Mat morpheus_r = imread("data/MorpheusR.jpg", 0);
    Size image_size_l = morpheus_l.size(); 
    // Size image_size_r = morpheus_r.size();
    Mat R1, R2, P1, P2, Q;
    stereoRectify(cam_matrix_l, noArray(), cam_matrix_r, 
                    noArray(), image_size_l, R, T, R1, R2, P1, P2, Q, 0);
    bool isVerticalStereo = false;
    isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    Mat l1, l2, r1, r2, imgl, imgr;
    initUndistortRectifyMap(cam_matrix_l, noArray(), R1, P1, image_size_l, CV_16SC2, l1, l2);
    initUndistortRectifyMap(cam_matrix_r, noArray(), R2, P2, image_size_l, CV_16SC2, r1, r2);
    Mat img_r, img_l;
    remap(morpheus_l, img_l, l1, l2, INTER_LINEAR);
    remap(morpheus_r, img_r, r1, r2, INTER_LINEAR);

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

    imshow("rectified_pair", pair);
    imwrite("pair.jpg", pair);
    
    imwrite("imgl.jpg", img_l);
    imwrite("imgr.jpg", img_r);
    while((char)waitKey(0) != 27);
    cria_mapas_morpheus(img_l, img_r, params);
    estima_medidas_morpheus(Q, params);
}

void estima_medidas_morpheus(Mat Q, parameters pars)
{
    Mat img = imread("morpheus_depth.png", -1);
    namedWindow("Req3", WINDOW_NORMAL);
    resizeWindow("Req3", 1000,  1000);
    while(1) {
        setMouseCallback("Req3", on_mouse, &pars);
        imshow( "Req3", img);
        if((char)waitKey(55) == 27) break;
    }
    Mat out;
    reprojectImageTo3D(img, out, Q);
   
    // Mat pt_1(4, 1, CV_64FC1), pt_2(4, 1, CV_64FC1);
    // for(int i = 0; i < pars.cliques.size(); i+=2)
    // {
    //     pt_1.at<double>(0,0) = (double)pars.cliques[i].first;
    //     pt_1.at<double>(1,0) = (double)pars.cliques[i].second;
    //     pt_1.at<double>(2,0) = img.at<double>(pars.cliques[i].second, pars.cliques[i].first);
    //     pt_1.at<double>(3,0) = 1;

    //     pt_2.at<double>(0,0) = (double)pars.cliques[i+1].first;
    //     pt_2.at<double>(1,0) = (double)pars.cliques[i+1].second;
    //     pt_2.at<double>(2,0) = img.at<double>(pars.cliques[i+1].second, pars.cliques[i+1].first);
    //     pt_2.at<double>(3,0) = 1;
    //     pt_1 = Q * pt_1;
    //     pt_2 = Q * pt_2;
    //     pt_1 = pt_1 / pt_1.at<double>(3, 0);
    //     pt_2 = pt_2 / pt_2.at<double>(3, 0);
    //     Mat dst_1, dst_2;
    //     pt_1.rowRange(0, 3).copyTo(dst_1);
    //     pt_2.rowRange(0, 3).copyTo(dst_2);
    //     cout << dst_1 << endl << dst_2 << endl;

    //     if(i == 0) cout << "largura em mm(?): " << norm(dst_1, dst_2, NORM_L2) << endl;
    //     else if(i == 2) cout << "altura em mm(?): " << norm(dst_1, dst_2, NORM_L2) << endl;
    //     else if(i == 4) cout << "profundidade em mm(?): " << norm(dst_1, dst_2, NORM_L2) << endl;
    // }

    for(int i = 0; i < pars.cliques.size(); i+=2)
    {
        Vec3b pt1 = out.at<Vec3b>(pars.cliques[i].second, pars.cliques[i].first);
        Vec3b pt2 = out.at<Vec3b>(pars.cliques[i+1].second, pars.cliques[i+1].first);
        if(i == 0) cout << "largura em mm: " << norm(pt1, pt2, NORM_L2) << endl;
        else if(i == 2) cout << "altura em mm: " << norm(pt1, pt2, NORM_L2) << endl;
        else if(i == 4) cout << "profundidade em mm: " << norm(pt1, pt2, NORM_L2) << endl;
    }
}

void cria_mapas_morpheus(Mat left_2, Mat right_2, parameters params) 
{
    Mat left, right;
    left_2.convertTo(left, CV_8U);
    right_2.convertTo(right, CV_8U);
    
    Ptr<DisparityWLSFilter> wls_filter;
    Mat left_for_matcher, right_for_matcher;
    Mat left_disp,right_disp;
    Mat filtered_disp;
    right_for_matcher = left.clone();
    left_for_matcher = right.clone();
    Mat conf_map = Mat(left.rows, left.cols, CV_8U);
    conf_map = Scalar(255);
    double matching_time, filtering_time;

    Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,params.max_disp,params.wsize);
    left_matcher->setP1(24*params.wsize*params.wsize);
    left_matcher->setP2(96*params.wsize*params.wsize);
    left_matcher->setPreFilterCap(63);
    left_matcher->setMode(StereoSGBM::MODE_SGBM);
    wls_filter = createDisparityWLSFilter(left_matcher);
    Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

    matching_time = (double)getTickCount();
    left_matcher->compute(left_for_matcher, right_for_matcher,left_disp);
    right_matcher->compute(right_for_matcher,left_for_matcher, right_disp);
    matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();

    wls_filter->setLambda(params.lambda);
    wls_filter->setSigmaColor(params.sigma);
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

    getDisparityVis(left_disp,raw_disp_vis,params.vis_mult);
    namedWindow("raw disparity", WINDOW_AUTOSIZE);
    imshow("raw disparity", raw_disp_vis);
    Mat disparity = left_disp.clone();
    double mini, maxi;
    minMaxIdx(left_disp, &mini, &maxi);
    disparity = (disparity - mini) * (255/(maxi - mini));
    disparity.convertTo(disparity, CV_8U);
    imshow("Mapa de disparidade", disparity);
    imwrite("morpheus_disp.png", raw_disp_vis);
    waitKey(0);

    Mat filtered_disp_vis;
    // imwrite("estima_morpheus.png", filtered_disp);
    getDisparityVis(filtered_disp, filtered_disp_vis, params.vis_mult);
    namedWindow("Mapa de profundidade", WINDOW_AUTOSIZE);
    imshow("Mapa de profundidade", filtered_disp_vis);
    imwrite("morpheus_depth.png", filtered_disp_vis);
    waitKey(0);
}