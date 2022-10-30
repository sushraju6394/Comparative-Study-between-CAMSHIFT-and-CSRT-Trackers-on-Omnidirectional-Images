#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <ctype.h>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <fstream>
#include <math.h>
#include <Windows.h>
#include "opencv2/video/background_segm.hpp"
#include <ctype.h>
#include <chrono>
using namespace cv;
using namespace std;
Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 10, vmax = 256, smin = 30;
Mat  foregroundMask, backgroundImage, foregroundImg;
Mat roi_BS;
Rect trackWindow1;
Mat hist1;
float e1,e2,t;
// User draws box around object to track. This triggers CAMShift to start tracking
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
		selection &= Rect(0, 0, image.cols, image.rows);//SELECTION IS IN THE FORM OF RECTANGLE
	}
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		break;
	case EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			trackObject = -1;   // Set up CAMShift properties in main() loop
		break;
	}
}
int main()
{

	//clock_t tStart = clock();
	auto start = chrono::steady_clock::now();
	e1 = getTickCount();//start time
	int count = 0;
	int correct = 0;
	int wrong = 0;
	string filename = "F:/CPictures/walking/Record_%05d.png";
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
	VideoCapture capture(filename);
	double fps = capture.get(CAP_PROP_FPS);
	cout << "Frames per second using capture.get(CV_CAP_PROP_FPS) : " << fps << endl;
	if (!capture.isOpened())
	{
		//error in opening the video input
		cerr << "Unable to open file!" << endl;
		return 0;
	}

	Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;
	namedWindow("Histogram", 0);
	namedWindow("CamShift Demo", 0);
	setMouseCallback("CamShift Demo", onMouse, 0);
	createTrackbar("Vmin", "CamShift Demo", &vmin, 256, 0);
	createTrackbar("Vmax", "CamShift Demo", &vmax, 256, 0);
	createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);
	Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;// returns 0 array of specified size and type
	bool paused = false;
	for (size_t i = 0; i < filename.size(); i++)
	{
		
		count = count + 1;
		cout << "frame count is" << count << endl;

		if (!paused)
		{
			capture >> frame;
			if (frame.empty())
				break;
		}
		frame.copyTo(image);
		if (!paused)
		{
			cvtColor(image, hsv, COLOR_BGR2HSV);
			if (trackObject)
			{
				int _vmin = vmin, _vmax = vmax;
				inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
					Scalar(180, 256, MAX(_vmin, _vmax)), mask);
				int ch[] = { 0, 0 };
				hue.create(hsv.size(), hsv.depth());
				mixChannels(&hsv, 1, &hue, 1, ch, 1);
				if (trackObject < 0)
				{
					// Object has been selected by user, set up CAMShift search properties once
					Mat roi(hue, selection), maskroi(mask, selection);
					calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
					normalize(hist, hist, 0, 255, NORM_MINMAX);
					trackWindow = selection;//SELECTING THE WINDOW FROM MOUSE CLICK
					trackObject = 1; // Don't set up again, unless user selects new ROI
					histimg = Scalar::all(0);//VARIABLE IN MAT FORM WHICH IS INITIALISED TO 0
					int binW = histimg.cols / hsize;//BIN WIDTH
					Mat buf(1, hsize, CV_8UC3);//BUF IS A TEMP VARIABLE
					for (int i = 0; i < hsize; i++)
						buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);
					cvtColor(buf, buf, COLOR_HSV2BGR);//CONVERT HSV BACK TO BGR
					for (int i = 0; i < hsize; i++)//DRAW HISTOGRAM IN RECT FORM
					{
						int val = saturate_cast<int>(hist.at<float>(i) * histimg.rows / 255);
						rectangle(histimg, Point(i * binW, histimg.rows),
							Point((i + 1) * binW, histimg.rows - val),
							Scalar(buf.at<Vec3b>(i)), -1, 8);
					}
				}
				// Perform camshift

				calcBackProject(&hue, 1, 0, hist, backproj, &phranges);//HIST IS THE OUTPUT MAT WHICH CONTAINS HISTOGRAM VALUES
				backproj &= mask;//DESTINATION BACK PROJ ARRAY
				RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
				trackWindow1 = trackBox.boundingRect();//current window
				trackWindow1 = selection;
				Mat roi1(hue, trackWindow);
				//cout << "trackWindow is" << trackWindow << endl;
				//Scalar avg_roi1 = mean(roi1);
				//cout << "avg value of roi1 is" << avg_roi1 << endl;
				calcHist(&roi1, 1, 0, Mat(), hist1, 1, &hsize, &phranges);
				normalize(hist1, hist1, 0, 255, NORM_MINMAX);
				double val = compareHist(hist, hist1, HISTCMP_BHATTACHARYYA);//find BC value between target and candidate histograms
				cout << "the bhattacharyya coefficient is " << val << endl;
				trackWindow1.width = 65;//Fixing the size of trackwindow
				trackWindow1.height = 55;
				trackWindow = trackWindow1;//update roi
				if (trackWindow.area() <= 1)
				{
					int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
					trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
						trackWindow.x + r, trackWindow.y + r) &
						Rect(0, 0, cols, rows);//SELECTING TOP LEFT X,Y AND BOTTOM RIGHT X,Y//
				}
				if (backprojMode)
					cvtColor(backproj, image, COLOR_GRAY2BGR);
				ellipse(image, trackBox, Scalar(0, 0, 255), 3, LINE_AA);
				// Background subtraction on the original frame//
				if (foregroundMask.empty())
				{
					foregroundMask.create(Size(640, 480), frame.type());
				}
				// compute foreground mask 8 bit image
				// -1 is parameter that chose automatically your learning rate
				bg_model->apply(frame, foregroundMask, true ? -1 : 0);
				// smooth the mask to reduce noise in image
				GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
				// threshold mask to saturate at black and white values
				threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
				//Background Subtractor;
				// create black foreground image
				foregroundImg = Scalar::all(0);
				// Copy source image to foreground image only in area with white mask
				frame.copyTo(foregroundImg, foregroundMask);
				//Get background image
				bg_model->getBackgroundImage(backgroundImage);
				// Show the results
				rectangle(foregroundMask, trackWindow, Scalar(255), 2, 8, 0);
				roi_BS = foregroundMask(trackWindow);
				float a = countNonZero(roi_BS);
				cout << "no of white pixels in track_window is" << a << endl;
				double percentage = (a / (65 * 55) * 100);
				cout << "the percentage of white pixels is" << percentage << endl;
				if (percentage > 30)
				{
					correct = correct + 1;
					cout << "Tracking done" << endl;

				}
				else
				{
					wrong = wrong + 1;
					cout << "Tracking not done" << endl;
				}
				namedWindow("foreground mask TRACKING", WINDOW_NORMAL);
				resizeWindow("foreground mask TRACKING", 1680, 1680);
				imshow("foreground mask TRACKING", foregroundMask);
				namedWindow("camshift", WINDOW_NORMAL);
				resizeWindow("frame", 1680, 1680);
				imshow("camshift", frame);//displays the output of camshift tracker
				cout << "The total number of tracked frames is" << correct << endl;
				cout << "The total number of not tracked frames is" << wrong << endl;
				//int keyboard = waitKey(30);
				//if (keyboard == 'q' || keyboard == 27 ||)
				cout << "The efficiency of camshift tracker is: " << (float(float(correct) / float(correct + wrong)) * 100) << endl;
			}
		}
		else if (trackObject < 0)
			paused = false;
		if (selectObject && selection.width > 0 && selection.height > 0)
		{
			Mat roi(image, selection);
			bitwise_not(roi, roi);
		}
		imshow("CamShift Demo", image);
		imshow("Histogram", histimg);
		char c = (char)waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'b':
			backprojMode = !backprojMode;
			break;
		case 'c':
			trackObject = 0;
			histimg = Scalar::all(0);
			break;
		case 'h':
			showHist = !showHist;
			if (!showHist)
				destroyWindow("Histogram");
			else
				namedWindow("Histogram", 1);
			break;
		case 'p':
			paused = !paused;
			break;
		default:
			;
		}
		e2 = getTickCount();
		t = (e2 - e1) / getTickFrequency();
		cout << "the execution time is" << t << endl;
		//printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		auto end = chrono::steady_clock::now();
		auto diff = end - start;
		cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
	}//end of for loop
	return 0;
}
