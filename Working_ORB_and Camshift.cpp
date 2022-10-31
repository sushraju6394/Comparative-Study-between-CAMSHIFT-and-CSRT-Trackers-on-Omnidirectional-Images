#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using namespace cv;
using namespace std;
//Global variables
Mat roi;//Global
vector<KeyPoint> kpts1;
Rect trackwindow;
Rect track_window(854, 487, 58, 92);

int main()
{
	string filename = "F:/CPictures/walking/Record_%05d.png";//pass the frames from folder
	//"F:/CPictures/White_floor/Record_%05d.png";
	VideoCapture capture(filename);//capture converts frames to video
	if (!capture.isOpened())
	{
		//error in opening the video input
		cerr << "Unable to open file!" << endl;
		return 0;
	}
	int framecount = 24;

	for (int frames = 0; frames < framecount; frames++)//frame by frame
	{


		Mat frame, hsv_roi;//initialisng,all images are stored in Matrix form
		Mat mask;
		// take first frame of the video
		capture >> frame;//0 frame
		// setup initial location of window
		if (frames == 0)
		{

			Rect track_window(846, 321, 50, 50);//initial roi 
		}

		// face simply hardcoded the values
		// set up the ROI for tracking

		roi = frame(track_window);//roi gets updated

		Ptr<cv::ORB> orb_detector = cv::ORB::create(10000);

		Mat mask1(frame.size(), CV_8UC1, Scalar::all(0));
		mask1(track_window).setTo(Scalar::all(255));

		orb_detector->detect(frame, kpts1, mask1);

		Mat res;
		drawKeypoints(frame, kpts1, res, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
		//namedWindow("roi", WINDOW_NORMAL);
		//resizeWindow("frame", 1680, 1680);
		//imshow("roi", res);


		cout << "# Keypoints 1:   \t" << kpts1.size() << endl;
		cout << endl;

		cout << "frames count" << frames << endl;
		cvtColor(roi, hsv_roi, COLOR_BGR2HSV);//converts BGR to HSV color space
		inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);
		float range_[] = { 0, 180 };
		const float* range[] = { range_ };
		Mat roi_hist;
		int histSize[] = { 180 };
		int channels[] = { 0 ,1 };
		calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);//find target histogram and store output in roi_hist

		normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);//normalise histogram
		// Setup the termination criteria, either 10 iteration or move by atleast 1 pt
		TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);//after how many iterations should the camshift tracker converge

		Mat hsv, dst;

		Point pt;//hardcoded values
		pt.x = 868;
		pt.y = 807;
		Point2f P(868, 807);
		int radMid = 288.5;
		circle(frame, pt, radMid, (0, 0, 255), 3, 8, 0);//drw fixed circle

		cvtColor(frame, hsv, COLOR_BGR2HSV);
		Mat dst1;
		//imshow("hsv color space", hsv);
		calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

		//one of the input to camshift,dst is in grey scale and is given to camshift
		//RotatedRect gives the rectangle box of variable rot_rect
		RotatedRect rot_rect = CamShift(dst, track_window, term_crit);
		
		//waitKey(2000);
		//lipse(frame, rot_rect, Scalar(0, 0, 255), 3, LINE_AA);
		trackwindow = rot_rect.boundingRect();//current
		
		//trackwindow.width = 65;
		//trackwindow.height = 55;
		//Size2f sizeRect = [40, 40];
		//cout << "the size is" << sizeRect << endl;

		track_window = trackwindow;//current is assigned to initial track window
		Point2f points[4];
		rot_rect.points(points);
		for (int i = 0; i < 4; i++)
			line(frame, points[i], points[(i + 1) % 4], 255, 2);

		namedWindow("frame", WINDOW_NORMAL);
		resizeWindow("frame", 1680, 1680);
		imshow("frame", res);//displays the output
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;



	}//end for//
	return 0;
}



