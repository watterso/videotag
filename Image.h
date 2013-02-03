#include <vector>
#include "opencv2/opencv.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;
class Image{
public:
    int identity;
    Mat frame;
    string path;
    Image(int i, Mat m, string p);
    ~Image();
};
