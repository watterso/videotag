#include "Image.h"


Image::Image(int i, Mat m, string p){
    identity = i;
    frame = m;
    path = p;
}
Image::~Image(){
    
}