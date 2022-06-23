#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;


/////////////////  Webcam face detection  //////////////////////

int main() {
    double scale = 2.0;
    CascadeClassifier faceCascade;
    faceCascade.load("Resources/haarcascade_frontalface_default.xml");

    
    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    for (;;)
    {
        Mat img;
        cap >> img;

        Mat grayscale;
        cvtColor(img, grayscale, COLOR_BGR2GRAY);
        resize(grayscale, grayscale, Size(grayscale.size().width / scale, grayscale.size().height / scale));

        if (faceCascade.empty())
        {
            cout << "XML file not loaded" << endl;
        }

        vector<Rect>faces;
        faceCascade.detectMultiScale(grayscale, faces, 1.1, 3, 0);

        for (int i = 0; i < faces.size(); i++)
        {
            rectangle(grayscale, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
        }

        imshow("Webcam", grayscale);

        if (waitKey(30) >= 0)
            break;
    }

    
    

    
    
}

