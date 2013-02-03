#include <vector>
#include "opencv2/opencv.hpp"
#include "Image.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

string people[7] ={"Tyler", "Simon", "Ricky", "Evan", "George", "Sean", "Harrison"};

//http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html#eigenfaces
static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    int i = 0;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat temp = imread(path, 0);
            images.push_back(temp);
            labels.push_back(atoi(classlabel.c_str()));
            cout<<i<<": "<<path<<" Dimens: "<<temp.size().width<<" x "<<temp.size().height<<endl;

        }
        
        i++;
    }
}

String NumberToString ( int Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}

int getdir (string dir, vector<string> &files){
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

int load = 0;
String face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
int main(int argc, const char *argv[]){
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if (argc < 3) {
        cout << "usage: " << argv[0] << " <csv.ext data set> <image directory> [facerecognizerYAML 1 - load 2-load/save] " << endl;
        exit(1);
    }
	if(argc==4){
		load =atoi(argv[3]);
		load--;
		cout<<"load: "<<load<<endl;
	}
    string dir = string(argv[2]);       //get the list of input images
    vector<string> files = vector<string>();
    getdir(dir,files);
    vector<Image> input = vector<Image>();
    for (int i =2; i<files.size(); i++) {	//skip '.' and '..'
		Mat temp1 = imread(dir+"/"+files[i],0);
		//cout<< files[i]<<" | Dimens: "<<temp1.size().width<<"x"<<temp1.size().height<<endl;
        input.push_back(Image(-1, temp1, files[i]));   //get vector of data in Image objects
    }
    
    string fn_csv = string(argv[1]);
    vector<Mat> images; //vector for faces
    vector<int> labels; //vector for labels
    //mooch dat csv implementation
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    if(images.size() <= 1) {
        string error_message = "Need at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    
    Size temp(0,0);
    for(int i = 0; i<images.size(); i++){           //calculate the largest dimensions for scaling up
        if(images[i].size().width>temp.width){
            temp.width = images[i].size().width;
        }
        if(images[i].size().height>temp.height){
            temp.height = images[i].size().height;
        }
    }
    
    for(int i = 0; i<input.size(); i++){           //calculate the largest dimensions for scaling up
        if(input[i].frame.size().width>temp.width){
            temp.width = input[i].frame.size().width;
        }
        if(input[i].frame.size().height>temp.height){
            temp.height = input[i].frame.size().height;
        }
    }
    cout<<"Dimens: "<<temp.width<<"x"<<temp.height<<endl;
    //resize the images so they are all teh sane size
    for(int i = 0; i<images.size(); i++){
        resize(images[i],images[i],temp);
    }
    for(int i = 0; i<input.size(); i++){
        resize(input[i].frame,input[i].frame,temp);
    }
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer(10, 15000); //create the trainer model (#eigen, threshold)
	if(load==1){
		model->load("model.yaml");
	}
	int maxLabel = 0;
	for (int i =0; i<labels.size(); i++) {
		if(labels[i]>maxLabel){
			maxLabel = labels[i];
		}
	}
	maxLabel++; //get the next highest
	cout<<images.size()<<" | "<<labels.size()<<endl;
    model->train(images, labels);
	vector<Mat> newFaces = vector<Mat>();
	vector<Mat> imgUpdate = vector<Mat>();
	vector<int> labelUpdate = vector<int>();
	for(int i = 0; i<input.size(); i++){
		double confidence = 0.0;
		model->predict(input[i].frame,input[i].identity, confidence);
		cout <<"Predicted class for "<< input[i].path<<" = "<< input[i].identity<< " ("<<confidence<<") "<<endl;
		if(confidence<=8000){
		cout <<"Predicted class for "<< input[i].path<<" = "<< input[i].identity<< " ("<<confidence<<") "<<endl;
			imgUpdate.push_back(input[i].frame);
			labelUpdate.push_back(input[i].identity);
		}else{
			//check if it is a new face
			Image temp1 = input[input.size()-1];
			input[input.size()-1] = input[i];
			input[i] = temp1;					//get rid of input[i]
			temp1 = input[input.size()-1];
			vector<Rect> faces;
			Mat grayFrame =  norm_0_255( temp1.frame);
			equalizeHist( grayFrame, grayFrame );
			face_cascade.detectMultiScale( grayFrame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
			if(faces.size()>0){
				newFaces.push_back(temp1.frame);
			}
			i--;
			input.pop_back();
		}
    }
	
	//programatically add 'newFaces' here
	

	if(imgUpdate.size()<0){
		cout<<imgUpdate.size()<<" | "<<labelUpdate.size()<<endl;
		model->train(imgUpdate, labelUpdate);
	}
	
	model->save("model.yaml");
	if(load>0){
		cout<<"save: "<<load<<endl;
	}
}





