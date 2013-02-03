#include "opencv2/opencv.hpp"
#include <CMath>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

void findAndShow(Mat frame);
void findRightEye(Mat frame);
void findLeftEye(Mat frame);
void findMouth(Mat frame);
void findNose(Mat frame);
void findFaces(Mat frame);

String NumberToString ( int Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    
    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

String cascadeClassifier [5]={"haarcascades/haarcascade_mcs_lefteye.xml","haarcascades/haarcascade_mcs_righteye.xml", "haarcascades/haarcascade_mcs_nose.xml", "haarcascades/haarcascade_mcs_mouth.xml","haarcascades/haarcascade_mcs_mouth.xml"};
int ind =0;

vector<Point> *mouths = new vector<Point>(); //vector of possible mouth locations
vector<Point> *nosess = new vector<Point>();  //vector of possible nose locations
vector<Point> *lefts = new vector<Point>();  //vector of possible left eye locations
vector<Point> *rights = new vector<Point>(); //vector of possible right eye locations
int frameNo = 0;
double FPS = 0;
int main(int, char**)
{   string dir = string("out");
    vector<string> files = vector<string>();
    
    getdir(dir,files);
    namedWindow("image",1);
    for(int i = 2; i<files.size(); i++){    //[0]='.' ; [1]='..' skip it!
        cout << files[i] << endl;
        Mat temp = imread("out/"+files[i], CV_LOAD_IMAGE_COLOR);
        findAndShow(temp);
        
        Point tempMouth(0,0);
        for(int j =0; j<mouths->size(); j++){
            int diff = abs(mouths->at(j).x-(temp.cols/2));      //find mouth closest to middle of image
            int diffS = abs(tempMouth.x-(temp.cols/2));
            if(diff<diffS){
                tempMouth = mouths->at(j);
            }
        }
        
        
        Point tempNose(0,0);
        for(int j =0; j<nosess->size(); j++){
            int diff = abs(nosess->at(j).x-(temp.cols/2));      //find nose closest to middle of image
            int diffS = abs(tempNose.x-(temp.cols/2));
            if(diff<diffS){
                tempNose = nosess->at(j);
            }
        }
        
        double dX  = tempNose.x-tempMouth.x;
        double dY = tempNose.y-tempMouth.y;
        cout << dX << " | "<< dY<<endl;
        
        double left = -dX/dY + tempMouth.y;
        double right = -dX/dY*temp.cols+tempMouth.y;
        Point furthestRight(temp.cols,right);
        Point furthestLeft(1,left);
        line(temp,furthestLeft,furthestRight,Scalar( 0, 255, 0 ));
        cout << left << " | "<< right<<endl;
        
        
        
        
        tempNose.y = 0;
        tempMouth.y=temp.rows;                                  //draw last because causing the points to be at the extremes messes up the math
        line(temp,tempMouth,tempNose,Scalar( 255, 255, 255 ));
        /*Point tempLefts(0,0);
        for (int j = 0; j<lefts->size(); j++) {
            int diff = abs(lefts->at(j).x-(temp.cols/4));
        }*/
        
        imshow( "image", temp);
        waitKey(0);
        mouths->clear();
        nosess->clear();
        lefts->clear();
        rights->clear();
    }
    /* VideoCapture cap("test.mp4");
    namedWindow("image",1);
    if(!cap.isOpened())  // check if we succeeded
    return -1;
    CvSize frame_size;
	frame_size.height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	frame_size.width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    FPS = cap.get(CV_CAP_PROP_FPS);
	double FOURCC = cap.get(CV_CAP_PROP_FOURCC);
    cout << "Dimen: "<<frame_size.width << "x" << frame_size.height << endl;
    cout << "FPS:  " << FPS << endl;
    
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        if(frameNo%((int)FPS/4)==0){
            //cout << "Current Frame: " << frameNo << endl;
            findFaces(frame);
            imshow( "image", frame );
            //waitKey(1) >= 0;
        }
        frameNo++;
        if(waitKey(30) >= 0) break;
    }*/
}

void findAndShow( Mat frame )
{
    findLeftEye(frame);
    findRightEye(frame);
    findNose(frame);
    findMouth(frame);
}
void findFaces(Mat frame){
    vector<Rect> faces;
    Mat grayFrame;
    cvtColor( frame, grayFrame, CV_BGR2GRAY );
    equalizeHist( grayFrame, grayFrame );
    
    CascadeClassifier cctemp;
    if( !cctemp.load( cascadeClassifier[4] ) )
	{
		std::cout<<"Error loading face cascade\n";
        exit(-1);
	}
    
    cctemp.detectMultiScale( grayFrame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    for(int i =0; i<faces.size(); i++){
        cout<<"HERE "<< i <<endl;
        Mat region = frame(faces[i]);   //get the 'Mat' for the face region
        findAndShow(region);
    }
}
void findLeftEye(Mat frame){
    Mat grayFrame;
    vector<Rect> lEye;
    
    cvtColor( frame, grayFrame, CV_BGR2GRAY );
    equalizeHist( grayFrame, grayFrame );           //grayscale
    
    CascadeClassifier cctemp;
    if( !cctemp.load( cascadeClassifier[0] ) )
	{
		std::cout<<"Error loading left eye cascade\n";
        exit(-1);
	}
    cctemp.detectMultiScale(frame,lEye,1.1,2,0);
    if(lEye.size()==0)
		return;
	
    for(int i=0;i<lEye.size();i++)
	{
        if(lEye[i].y>frame.rows/2 ||lEye[i].y<frame.rows/4 )continue; //eye can't be in bottom half of image
        if(lEye[i].x>frame.cols/2)continue; //left eyes won't be on the right
		Point center( lEye[i].x + lEye[i].width*0.5, lEye[i].y + lEye[i].height*0.5 );
        lefts->push_back(center);
        ellipse( frame, center, Size( lEye[i].width*0.5, lEye[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );
	}
}
void findRightEye(Mat frame){
    Mat grayFrame;
    vector<Rect> rEye;
    
    cvtColor( frame, grayFrame, CV_BGR2GRAY );
    equalizeHist( grayFrame, grayFrame );           //grayscale
    
    CascadeClassifier cctemp;
    if( !cctemp.load( cascadeClassifier[1] ) )
	{
		std::cout<<"Error loading right eye cascade\n";
        exit(-1);
	}
    cctemp.detectMultiScale(frame,rEye,1.1,2,0);
    if(rEye.size()==0)
		return;
	
    for(int i=0;i<rEye.size();i++)
	{
        if(rEye[i].y>frame.rows/2 ||rEye[i].y<frame.rows/4 )continue; //eye can't be in bottom half of image
        if(rEye[i].x<frame.cols/2)continue; //  right eyes won't be on the left
		Point center( rEye[i].x + rEye[i].width*0.5, rEye[i].y + rEye[i].height*0.5 );
        rights->push_back(center);
        ellipse( frame, center, Size( rEye[i].width*0.5, rEye[i].height*0.5), 0, 0, 360, Scalar( 0, 0, 255 ), 4, 8, 0 );
	}
}
void findNose(Mat frame){
    Mat grayFrame;
    vector<Rect> noses;
    
    cvtColor( frame, grayFrame, CV_BGR2GRAY );
    equalizeHist( grayFrame, grayFrame );           //grayscale
    
    CascadeClassifier cctemp;
    if( !cctemp.load( cascadeClassifier[2] ) )
	{
		std::cout<<"Error loading right eye cascade\n";
        exit(-1);
	}
    cctemp.detectMultiScale(frame,noses,1.1,2,0);
    if(noses.size()==0)
        //cout<<"No noses found"<<endl;
		return;
	
    for(int i=0;i<noses.size();i++)
	{
        if(noses[i].y<frame.rows/4 || noses[i].y>3*frame.rows/4)continue; //no noses in the top quarter or bottom
		Point center( noses[i].x + noses[i].width*0.5, noses[i].y + noses[i].height*0.5 );
        nosess->push_back(center);
        ellipse( frame, center, Size( noses[i].width*0.5, noses[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );
	}

}
void findMouth(Mat frame){
    Mat grayFrame;
    vector<Rect> mouth;
    
    cvtColor( frame, grayFrame, CV_BGR2GRAY );
    equalizeHist( grayFrame, grayFrame );           //grayscale
    
    CascadeClassifier cctemp;
    if( !cctemp.load( cascadeClassifier[3] ) )
	{
		std::cout<<"Error loading right eye cascade\n";
        exit(-1);
	}
    cctemp.detectMultiScale(frame,mouth,1.1,2,0);
    if(mouth.size()==0)
        //cout<<"No mouth found"<<endl;
		return;
	
    for(int i=0;i<mouth.size();i++)
	{
        if(mouth[i].y<frame.rows/2)continue; //mouth will be in bottom half of photo
		Point center( mouth[i].x + mouth[i].width*0.5, mouth[i].y + mouth[i].height*0.5 );
        mouths->push_back(center);
        ellipse( frame, center, Size( mouth[i].width*0.5, mouth[i].height*0.5), 0, 0, 360, Scalar( 120, 120, 120 ), 4, 8, 0 );
	}
}
