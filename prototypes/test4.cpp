#include <vector>
#include "opencv2/opencv.hpp"
#include <iostream>


using namespace cv;
using namespace std;

string NumberToString ( int Number )
{
    ostringstream ss;
    ss << Number;
    return ss.str();
}
string people[7] ={"Tyler", "Simon", "Ricky", "Evan", "George", "Sean", "Harrison"};
int main(int argc, const char *argv[]) {
    if (argc != 3) {
        cout << "usage: " << argv[0] << " <invideo> <outdir>" << endl;
	}
	string fn_haar = "haarcascades/haarcascade_frontalface_alt.xml";
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer(10, 12000);
	model->load("../model.yaml");
	CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(argv[1]);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        return -1;
    }
	int ind = 100;
    // Holds the current frame from the Video device:
    Mat frame;
    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            //
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
            //
            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
            // face you have just found:
            Mat face_resized;
            cv::resize(face, face_resized, Size(389, 389), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:
            string box_text = format("Prediction = (%d)", prediction);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
		}
		string dir =argv[2];
		imwrite(dir+"/"+NumberToString(ind)+".jpg", original);
		ind++;
    }
    return 0;
}