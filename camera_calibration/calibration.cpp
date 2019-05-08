#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define DMIN 30
#define DMED 44
#define DMAX 58

struct Imagem {
	Mat img;
    int x1, y1, x2, y2;
    int cliques = 0;
    bool clique = false;
    bool primeiro_clique = false;
    bool segundo_clique = false;
    bool cinza = false;
};

void help(char **argv) { 
    cout << "\n\n"
         << "Reading a chessboard’s width and height,\n"
         << "              reading and collecting the requested number of views,\n" 
         << "              and calibrating the camera\n\n" 
         << "Call:\n" << argv[0] << " <board_width> <board_height> <number_of_boards> <if_video,_delay_between_frame_capture> <image_scaling_factor>\n\n"
         << "Example:\n" << argv[0] << " 8 6 24 500 0.5\n"
         << "-- to use the checkerboard8x6 provided\n\n"
         << " * First it reads in checker boards and calibrates itself\n" 
         << " * Then it saves and reloads the calibration matricies\n"
         << " * Then it creates an undistortion map and finally\n"
         << " * It displays an undistorted image\n"
         << endl;
}

void on_mouse(int event, int x, int y, int flags, void* userdata) {
    Imagem* img = (Imagem*) userdata;
    if (event == EVENT_LBUTTONDOWN) {
        if(img->primeiro_clique) {
            img->segundo_clique = true;
            img->x2 = x;
            img->y2 = y;
            line(img->img, Point(img->x1,img->y1), Point(x, y), Scalar(), 3);
            printf("Comprimento: %.3f pixels\n", sqrt((img->x1 - x)*(img->x1 - x) + (img->y1 - y)*(img->y1 - y)));
            printf("Pixel 1 = (%d,%d)\nPixel 2 = (%d,%d)\n", img->x1, img->y1, x, y);
        } else {
            img->segundo_clique = false;
            img->x1 = x;
            img->y1 = y;
        }
        img->primeiro_clique ^= true;
        img->clique = true;
    } else {
        img->clique = false;
    } 
}

void object_size(int x1, int y1, int x2, int y2, int z, Mat camera_matrix) {
    Mat canto1(1, 3, CV_64F, Scalar());
    canto1.at<double>(0,0) = x1;
    canto1.at<double>(0,1) = y1;
    canto1.at<double>(0,2) = 1;
    
    Mat canto2(1, 3, CV_64F, Scalar());
    canto2.at<double>(0,0) = x2;
    canto2.at<double>(0,1) = y2;
    canto2.at<double>(0,2) = 1;
    
    canto1 = z*(camera_matrix.inv() * canto1.t());
    canto2 = z*(camera_matrix.inv() * canto2.t());
    double norma = norm(canto2, canto1, NORM_L2);
    printf("Altura/Largura em cm: %f\nEm milimetros: %f\n", norma, norma*10);
}


int main(int argc, char *argv[]) {
    int n_boards = 0;           // will be set by input list
    float image_sf = 0.5f;      // image scaling factor
    float delay = 1.f;
    int board_w = 0;
    int board_h = 0;

    if (argc < 4 || argc > 6) {
        cout << "\nERROR: Wrong number of input parameters\n";
        help(argv);
        return -1;
    }

    board_w = atoi(argv[1]);
    board_h = atoi(argv[2]);
    n_boards = atoi(argv[3]);

    if (argc > 4) {
        delay = atof(argv[4]);
    }
    if (argc > 5) {
        image_sf = atof(argv[5]);
    }

    int board_n = board_w * board_h;
    cv::Size board_sz = cv::Size(board_w, board_h);
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        cout << "\nCouldn't open the camera\n";
        help(argv);
        return -1;
    }

    // ALLOCATE STORAGE
    //
    vector<vector<cv::Point2f> > image_points;
    vector<vector<cv::Point3f> > object_points;

    // Capture corner views: loop until we've got n_boards successful
    // captures (all corners on the board are found).
    //
    double last_captured_timestamp = 0;
    cv::Size image_size;
    while (image_points.size() < (size_t)n_boards) {
        cv::Mat image0, image;
        capture >> image0;
        image_size = image0.size();
        cv::resize(image0, image, cv::Size(), image_sf, image_sf, cv::INTER_LINEAR);

        // Find the board
        //
        vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(image, board_sz, corners);

        // Draw it
        //
        drawChessboardCorners(image, board_sz, corners, found);

        // If we got a good board, add it to our data
        //
        double timestamp = static_cast<double>(clock()) / CLOCKS_PER_SEC;
        if (found && timestamp - last_captured_timestamp > 1) {
            last_captured_timestamp = timestamp;
            image ^= cv::Scalar::all(255);
            cv::Mat mcorners(corners);

            // do not copy the data
            mcorners *= (1.0 / image_sf);

            // scale the corner coordinates
            image_points.push_back(corners);
            object_points.push_back(vector<cv::Point3f>());
            vector<cv::Point3f> &opts = object_points.back();

            opts.resize(board_n);
            for (int j = 0; j < board_n; j++) {
                opts[j] = cv::Point3f(static_cast<float>(j / board_w),
                                    static_cast<float>(j % board_w), 0.0f);
            }
            cout << "Collected our " << static_cast<uint>(image_points.size())
                << " of " << n_boards << " needed chessboard images\n" << endl;
        }
        cv::imshow("Calibration", image);

        // show in color if we did collect the image
        if ((cv::waitKey(30) & 255) == 27)
            return -1;
    }

    // END COLLECTION WHILE LOOP.
    destroyWindow("Calibration");
    cout << "\n\n*** CALIBRATING THE CAMERA...\n" << endl;

    // CALIBRATE THE CAMERA!
    //
    Mat intrinsic_matrix, distortion_coeffs;
    Mat tvec, rvec, rmat;
    double err = calibrateCamera(
        object_points, image_points, image_size, intrinsic_matrix,
        distortion_coeffs, rvec, tvec,
        cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT);

    // SAVE THE INTRINSICS AND DISTORTIONS
    cout << " *** DONE!\n\nReprojection error is " << err
         << "\nStoring Intrinsics.xml and Distortions.xml files\n\n";
    cv::FileStorage fs("intrinsics.xml", cv::FileStorage::WRITE);
    fs << "image_width" << image_size.width << "image_height" << image_size.height
       << "camera_matrix" << intrinsic_matrix << "distortion_coefficients"
       << distortion_coeffs;
    fs.release();

    // EXAMPLE OF LOADING THESE MATRICES BACK IN:
    fs.open("intrinsics.xml", cv::FileStorage::READ);
    cout << "\nimage width: " << static_cast<int>(fs["image_width"]);
    cout << "\nimage height: " << static_cast<int>(fs["image_height"]);
    cv::Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coefficients"] >> distortion_coeffs_loaded;
    cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
    cout << "\ndistortion coefficients: " << distortion_coeffs_loaded << endl;

    // Build the undistort map which we will use for all
    // subsequent frames.
    //
    cv::Mat map1, map2;
    initUndistortRectifyMap(intrinsic_matrix_loaded, distortion_coeffs_loaded,
                              cv::Mat(), intrinsic_matrix_loaded, image_size,
                              CV_16SC2, map1, map2);


    cout << "Rvec" << rvec << endl << "Tvec" << tvec << endl;
    Imagem frame;
    Imagem frame_undis;
    namedWindow("Undistorted", WINDOW_AUTOSIZE);
    namedWindow("Raw", WINDOW_AUTOSIZE);
    for (;;) {
        capture >> frame.img;
        frame_undis.img = (frame.img).clone();
        if ((frame.img).empty())
            break;

        setMouseCallback("Raw", on_mouse, &frame);
        bool raw = !frame.clique && frame.segundo_clique;
        if(raw) 
            line(frame.img, Point(frame.x1,frame.y1), Point(frame.x2, frame.y2), Scalar(), 3);
        imshow("Raw", frame.img);


        remap(frame_undis.img, frame_undis.img, map1, map2, INTER_LINEAR,
                cv::BORDER_CONSTANT, Scalar());
        setMouseCallback("Undistorted", on_mouse, &frame_undis);
        bool undis = !frame_undis.clique && frame_undis.segundo_clique;
        if(undis)
            line(frame_undis.img, Point(frame_undis.x1,frame_undis.y1), 
                    Point(frame_undis.x2, frame_undis.y2), Scalar(), 3);
        imshow("Undistorted", frame_undis.img);

        char calcula = (char)waitKey(45);
        if(raw and calcula == '1')
            object_size(frame.x1, frame.y1, frame.x2, frame.y2, DMIN, intrinsic_matrix_loaded);
        else if(raw and calcula == '2')
            object_size(frame.x1, frame.y1, frame.x2, frame.y2, DMED, intrinsic_matrix_loaded);
        else if(raw and calcula == '3')
            object_size(frame.x1, frame.y1, frame.x2, frame.y2, DMAX, intrinsic_matrix_loaded);
        else if(undis and calcula == '4')
            object_size(frame_undis.x1, frame_undis.y1, frame_undis.x2, frame_undis.y2, DMIN, intrinsic_matrix_loaded);
        else if(undis and calcula == '5')
            object_size(frame_undis.x1, frame_undis.y1, frame_undis.x2, frame_undis.y2, DMED, intrinsic_matrix_loaded);
        else if(undis and calcula == '6')
            object_size(frame_undis.x1, frame_undis.y1, frame_undis.x2, frame_undis.y2, DMAX, intrinsic_matrix_loaded);
        else if(calcula == 27)
            break;
    }

    return 0;
}
