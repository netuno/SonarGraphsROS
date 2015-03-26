#ifndef SONAR_H
#define SONAR_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "Gaussian.h"

using namespace std;
using namespace cv;

class Sonar
{
private:
    static const int neighborRow[8];
    static const int neighborCol[8];

    Mat floodFillResult; // 2 channels 1xN matrix , channel 0 = row found , channel 2 = col found by floodfill search
    unsigned floodFillResultCount; // Pixel found count by floodfill search
    unsigned MRCol, MRow, MCRow, MCol;

    bool computeMaxPixel;
    unsigned maxPixel;

    float stdDevMultiply; // Number of times we multiply stdDev to work
    float mergeStdDevMultiply; // Number of times we multiply stdDev to detect merge

    bool doMerge;
    bool computeGassianAngle;

    uchar pixelThreshold; // Threshold for pixel search (start floodfill)
    uchar searchThreshold; // Threshold to floodfill search
    Mat searchMask; // Used by floodfill search
    Mat colorImg;

    bool drawPixelFound;

    vector<Gaussian> gaussians;

    void floodFill(Mat &img, unsigned lin, unsigned col);

    float calcGaussianAng(float x, float y,float dx, float dy,float Mx, float MXy , float My, float MYx);

    void calculateMeanAndStdDerivation(Mat &img, int row, int col, float &rowMean, float &colMean, float &pixelMean, float &rowStdDev, float &colStdDev, float &pixelStdDev, float &ang, unsigned &N);

    void createGaussian(Mat &img);

    void drawGaussians();

    bool intersec(Gaussian &a, Gaussian &b);

    void mergeGaussian(unsigned a, unsigned b);

    Gaussian pseudoMergeGaussian(Gaussian &ga, Gaussian &gb);

public:
    Sonar();

    void newImage(Mat img);

};

#endif // SONAR_H
