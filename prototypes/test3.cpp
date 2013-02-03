#include <vector>
#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <sstream>


using namespace cv;
using namespace std;




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


int main(int argc, const char *argv[]){
    if (argc < 3) {
        cout << "usage: " << argv[0] << " <csv.ext> <imagetopredict> " << endl;
        exit(1);
    }
    Mat testSample = imread(string(argv[2]),0);
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
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
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
    
    //resize the images so they are all teh sane size
    for(int i = 0; i<images.size(); i++){
        resize(images[i],images[i],temp);
    }
    cout<<"test yet to be resized"<<endl;
    imshow("etetet", testSample);
   
    waitKey(0);
    
    resize(testSample,testSample, temp);
    int height = temp.height; //store image height for later restore
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer(); //create the trainer model
    model->train(images, labels);
    
    int predictedLabel = model->predict(testSample);
    string result_message = format("Predicted class = %d", predictedLabel);
    cout << result_message << endl;
    
    //now get specific options
     model->set("threshold", 0.0); // predictions will return -1?
    predictedLabel = model->predict(testSample);
    cout << "Predicted class = " << predictedLabel << endl;
    
    Mat eigenvalues = model->getMat("eigenvalues");
    Mat W = model->getMat("eigenvectors");
    
    for (int i = 0; i < min(10, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;
        // get eigenvector #i
        Mat ev = W.col(i).clone();
        // Reshape to original size & normalize to [0...255] for imshow.
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // Show the image & apply a Jet colormap for better sensing.
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        imshow(format("%d", i), cgrayscale);
    }
    waitKey(0);
    
    return 0;

}





