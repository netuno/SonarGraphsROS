#include <iostream>

#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Sonar.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    char img_file_name[200];
    FILE *f_img_names = fopen("imgs.txt","r");

    cout << -90+180.0*atan2(1,-1)/M_PI << endl;
    cout << -90+180.0*atan2(1,1)/M_PI << endl;
    cout << -90+180.0*atan2(-1,1)/M_PI << endl;
    cout << -90+180.0*atan2(-1,-1)/M_PI << endl;
    cout << endl;
    cout << 180.0*atan2(1,1)/M_PI << endl;
    cout << 180.0*atan2(-1,1)/M_PI << endl;
    cout << 180.0*atan2(-1,-1)/M_PI << endl;
    cout << 180.0*atan2( 1,-1)/M_PI << endl;

    Sonar s;
    while( fscanf(f_img_names,"%s", img_file_name) != -1)
    {
        Mat img = imread(img_file_name);
        s.newImage(img);
        waitKey();
    }
    return 0;
}

