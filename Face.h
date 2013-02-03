#include <vector>
#include "opencv2/opencv.hpp"
#include <stdio.h>

using namespace cv;
using namespace std;
class Face{
public:
    int x;
    int y;
    int height;
    int width;
    int edges;
    int frame;
    String fileName;
    Face(Rect r, int e, int f, string s);
    ~Face();
};
