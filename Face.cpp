#include "Face.h"


Face::Face(Rect r, int e, int f, String s){
    x = r.x;
    y = r.y;
    width = r.width;
    height = r.height;
    edges = e;
    frame = f;
    fileName = s;
}
Face::~Face(){
    
}