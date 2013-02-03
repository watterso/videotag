#include <vector>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int, char**)
{
    Mat frame; //raw frame
    Mat back; // background image
    Mat fore; // foreground mask
    VideoCapture cap("http://lwsnb160-cam.cs.purdue.edu/mjpg/video.mjpg"); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    
    BackgroundSubtractorMOG2 bg;
    
    vector<vector<Point> > contours;
    
    namedWindow("Frame");
    cv::namedWindow("Background");
    
    for(;;)
    {
        cap >> frame;
        bg.operator()(frame,fore);
        bg.getBackgroundImage(back);
        
        erode(fore,fore,Mat());
        dilate(fore,fore,Mat());
        findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        drawContours(frame,contours,-1,Scalar(0,0,255),2);
        imshow("Frame",frame);
        imshow("Background",back);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}