#include <vector>
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <CMath>
#include "Face.h"

using namespace cv;
using namespace std;

void findAndSave(Mat frame);

String NumberToString ( int Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}

String face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
int ind =0;
vector<Face> *mFaces = new vector<Face>();
int frameNo = 0;
double FPS = 0;
string oot= ""; //output dir
int main(int argc, const char *argv[]){
    if (argc < 3) {
        cout << "usage: " << argv[0] << " <input video> <output directory> " << endl;
        exit(1);
    }
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   
    oot = argv[2];
    VideoCapture cap(argv[1]); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    
    CvSize frame_size;
	frame_size.height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	frame_size.width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    FPS = cap.get(CV_CAP_PROP_FPS);
	double FOURCC = cap.get(CV_CAP_PROP_FOURCC);
    cout << "Dimen: "<<frame_size.width << "x" << frame_size.height << endl;
    cout << "FPS:  " << FPS << endl;
    cout << "FOURCC:  " << FOURCC << endl;
    //namedWindow("circles",2);

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        if(frameNo%((int)FPS/4)==0){
            //cout << "Current Frame: " << frameNo << endl;
            findAndSave(frame);
            //waitKey(1) >= 0;
        }
        frameNo++;
        //if(waitKey(1) >= 0) break;
    }
    return 0;
}
void findAndSave(Mat frame){
    vector<Rect> faces;
    Mat grayFrame;
    cvtColor( frame, grayFrame, CV_BGR2GRAY );
    equalizeHist( grayFrame, grayFrame );
    
    face_cascade.detectMultiScale( grayFrame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    for(int i =0; i<faces.size(); i++){
        try {
            Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
            //ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
            if(faces[i].width<125 || faces[i].height<125){
                continue;
            }
            Mat edges;
            Canny(frame(faces[i]), edges, 0, 30);
            int edgeCount = countNonZero(edges);
            if(countNonZero(edges)<1000) {
                //cout << "not enough edges: "<< edgeCount<<endl;
                continue;
            }
            bool flag = false;
            for(int j = 0; j<mFaces->size(); j++){
                Face temp = mFaces->at(j);
                if(abs(temp.x-faces[i].x)>20 || abs(temp.y-faces[i].y)>20) continue; //if its not in the same area, check the next
                if(edgeCount<temp.edges){ flag=true; break;}//if it has less edges don't save
                if((frameNo-temp.frame)>FPS){
                    continue;   //its someone else
                }
                //it is same face but higher res
                String c =oot +"/"+ temp.fileName;
                imwrite(c,frame(faces[i]));         //overwrite
                //cout<< "overwrote: "<<c<<endl;
                flag = true;
            }
            if(flag)continue;
            /*cout << "--------------------" << ind << "------------------"<<endl;
            cout << "start: "<<faces[i].x << "x" << faces[i].y << endl;
            cout << "Dimen: "<<faces[i].width << "x" << faces[i].height << endl;
            cout << "-------------------" << endl;*/
            String c =oot+"/" + NumberToString(ind)+".jpg";
            mFaces->push_back(*(new Face(faces[i], edgeCount, frameNo, c)));
            imwrite(c,frame(faces[i]));
            ind++;
        }
        catch (int ex) {
            fprintf(stderr, "Exception saving image to jpg format: %d\n", ex);
            return;
        }
    }
    //imshow( "circles", frame );
}





