#include <vector>
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <algorithm>

using namespace cv;
using namespace std;

int main(int, char** args)
{
    
    Mat blur = imread(args[1], CV_LOAD_IMAGE_ANYDEPTH);
    
    Mat canBlur;
    
    Canny(blur, canBlur, 0, 30);
    
    cout << "edges: " << countNonZero(canBlur) << endl;
    
    /*namedWindow("blur",1);
    namedWindow("blur - canny",2);
    
    namedWindow("no blur",3);
    namedWindow("no blur - canny",4);
    
    
    imshow("blur", blur);
    imshow("blur - canny", canBlur);
    imshow("no blur", noBlur);
    imshow("no blur - canny", noCan);
    
    waitKey(0) >= 0;*/
}



