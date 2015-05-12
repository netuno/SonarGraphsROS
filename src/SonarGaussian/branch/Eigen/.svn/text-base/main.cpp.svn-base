#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define INFINITY 0x3f3f3f3f

using namespace std;
using namespace cv;

void printVType(Mat v)
{
    if(v.type() == CV_32F)
        cout << "CV_32F" << endl;
    else if(v.type() == CV_64F)
        cout << "CV_64F" << endl;
}

int main()
{
    Mat_<float> samples = (Mat_<float>(3, 4) << -1.0, -2.0, -3.6, -4.0,
                                                -1.0, -1.0, -1.0, -2.0,
                                                 5.0,6.0,4.0,5.0);

    cout << "Amostra: " << samples << endl;

    Mat cov, mu;
    cv::calcCovarMatrix(samples, cov, mu, CV_COVAR_NORMAL | CV_COVAR_COLS, CV_32F);

    cout << "cov: " << endl;
    cout << cov << endl;

    cout << "mu: " << endl;
    cout << mu << endl;

    Mat eval, evect;
    eigen(cov,true,eval, evect);

    printVType(eval);
    cout << "eval: " << eval << endl;
    printVType(evect);
    cout << "evect: " << evect << endl;

    float ang;
    ang = 180.0*atan2(evect.at<float>(0,0),evect.at<float>(0,1))/M_PI;

    cout << "ang: " << ang << endl;
    cout << "stdx: " << eval.at<float>(0,0) << endl;
    cout << "stdy: " << eval.at<float>(1,0) << endl;
    cout << "stdI: " << eval.at<float>(2,0) << endl;

    float finf = INFINITY;
    int iinf = INFINITY;

    cout << "finf: " << finf << " " << finf*finf << " " << finf*finf*finf << endl;
    cout << "iinf: " << iinf << " " << iinf*iinf << " " << iinf*iinf*iinf << endl;
    return 0;
}

// A matrix de covariancia é sensivel ao tamanho da amostra,
// por esse motivo, eu acho, os autovalores da matriz de covariancia
// não são equivalentes ao desvio padrão.
