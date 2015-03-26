#include "Sonar.h"
#include <iostream>
using namespace std;

typedef pair<unsigned, unsigned> pii;

const int Sonar::neighborRow[8] = { -1, -1, -1,  0,  0,  1, 1, 1};
const int Sonar::neighborCol[8] = { -1,  0,  1, -1,  1, -1, 0, 1};

void Sonar::floodFill(Mat &img, unsigned row, unsigned col)
{
    searchMask.at<uchar>(row,col) = 255;

    /* Do something */
    floodFillResult.at<ushort>(0,floodFillResultCount*3) = row;
    floodFillResult.at<ushort>(0,floodFillResultCount*3+1) = col;
    floodFillResult.at<ushort>(0,floodFillResultCount*3+2) = img.at<uchar>(row,col);
    floodFillResultCount++;

    if(MRow < row)
    {
        MRCol = col;
        MRow = row;
    }
    if(MCol < col)
    {
        MCol = col;
        MCRow = row;
    }

    for(unsigned neighbor = 0 ; neighbor < 8 ; neighbor++)
    {
        unsigned nextRow = row + neighborRow[neighbor],
                 nextCol = col + neighborCol[neighbor];

        if(nextRow >= img.rows && nextRow < 0 &&
           nextCol >= img.cols && nextCol < 0)
            continue;

        if(img.at<uchar>(nextRow , nextCol) >= searchThreshold
           && searchMask.at<uchar>(nextRow,nextCol) != 255)
        {
            floodFill(img, nextRow , nextCol);
        }
    }
}

float Sonar::calcGaussianAng(float x, float y, float dx, float dy, float Mx, float MXy, float My, float MYx)
{
    if(!computeGassianAngle)
        return 0.0;

    float ang;
    if(dy > dx)
    {
        ang = 180.0*atan2(dx,dy)/M_PI;
        if(MXy > y) // Change rotation orientation
            ang = -ang;

    }else// Change the rotatios axes
    {
        ang = 180.0*atan2(dy,dx)/M_PI;
        if(MYx < x) // Change rotation orientation
            ang = -ang;
    }

    return ang;
}

void Sonar::calculateMeanAndStdDerivation(Mat &img, int row, int col,
                                          float &rowMean, float &colMean, float &pixelMean,
                                          float &rowStdDev, float &colStdDev, float &pixelStdDev,
                                          float &ang, unsigned &N)
{
    // Clear old floodfill results
    if(floodFillResult.rows != 1 || floodFillResult.cols != img.rows * img.cols)
    {
        cout << "Realoc!!" << endl;
        floodFillResult = Mat(1, img.rows * img.cols, CV_16UC3, Scalar(0,0,0));
    }
    floodFillResultCount = 0;
    MCol = MRow = 0;
    MCRow = MRCol = 99999;

    floodFill(img,row , col);

    if(drawPixelFound)
    for(unsigned i = 0 ; i < floodFillResultCount ; i++)
    {
        // Here:
        //   loodFillResult.at<ushort>(0, k ) - Acess the row of K-th pixel found by floodfill search
        //   floodFillResult.at<ushort>(0, k+1 ) - Acess the col of K-th pixel found by floodfill search
        //   floodFillResult.at<ushort>(0, k+2 ) - Acess the pixel intensity of K-th pixel found by floodfill search
        /// @todo - Acessing uchar Mat::data[] is faster than at<type>()

        unsigned result = i*3,
                pixelCol = floodFillResult.at<ushort>(0,result+1)*3;

        colorImg.at<uchar>(floodFillResult.at<ushort>(0,result), pixelCol) = 0;
        colorImg.at<uchar>(floodFillResult.at<ushort>(0,result), pixelCol+1) = 255;
        colorImg.at<uchar>(floodFillResult.at<ushort>(0,result), pixelCol+2) = 0;
    }

    // Calculate mean and standard deviation
    Mat segmentResult = floodFillResult(Rect(0,0,floodFillResultCount-1,1));
    Scalar mean, dev;
    meanStdDev(segmentResult, mean, dev);

    rowMean = static_cast<float>(mean.val[0]);
    colMean = static_cast<float>(mean.val[1]);
    pixelMean = static_cast<float>(mean.val[2]);
    rowStdDev = static_cast<float>(dev.val[0]);
    colStdDev = static_cast<float>(dev.val[1]);
    pixelStdDev = static_cast<float>(dev.val[2]);
    N = floodFillResultCount;
    ang = calcGaussianAng(colMean,rowMean,colStdDev,rowStdDev,
                          MCol,MCRow,MRow,MRCol);

    /*
    if(rowStdDev > colStdDev)
    {
        ang = 180.0*atan2(colStdDev,rowStdDev)/M_PI;
        if(MCRow > rowMean) // Change rotation orientation
            ang = -ang;

    }else// Change the rotatios axes
    {
        ang = 180.0*atan2(rowStdDev,colStdDev)/M_PI;
        if(MRCol < colMean) // Change rotation orientation
            ang = -ang;
    }
    */
}

void Sonar::createGaussian(Mat &img)
{
    /* Initialize Mask of Visit */
    searchMask = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));

    /* Clear old Gaussians */
    gaussians.clear();

    /* Do some image transform */
    if(img.type() != CV_8UC3)
    {
        cvtColor(img, colorImg, CV_GRAY2BGR);
    }
    else
    {
        img.copyTo(colorImg);
        cvtColor(img,img,CV_BGR2GRAY);
    }

    float rowMean,colMean,pixelMean,ang,
          rowStdDev,colStdDev, pixelStdDev;

    /* Search for high intensity pixels */
    Gaussian g;
    bool merge;
    unsigned linJump = 5, colJump = 5,N;
    for(unsigned lin = 0 ; lin < img.rows ; lin+=linJump)
    {
        for(unsigned col = 0 ; col < img.cols ; col+=colJump)
        {
            if(searchMask.at<uchar>(lin,col) == 0 &&
               img.at<uchar>(lin,col) >= pixelThreshold)
            {
                calculateMeanAndStdDerivation(img, lin, col,
                                        rowMean,colMean,pixelMean,
                                        rowStdDev, colStdDev, pixelStdDev,ang,N);
                cout << "( " << rowMean << " , " << colMean << " , " << pixelMean <<
                ") ( " << rowStdDev << " , " << colStdDev <<" , " << pixelStdDev <<
                    ") (" << ang  << " , " << N << " )" << endl;


                g = Gaussian(colMean,rowMean,pixelMean,
                    colStdDev*stdDevMultiply,rowStdDev*stdDevMultiply,pixelStdDev,ang,N);

                merge = false;

                if(doMerge)
                for(unsigned i = 0 ; i < gaussians.size() ; i++)
                {
                    if(intersec(gaussians[i], g))
                    {
                        gaussians[i] = pseudoMergeGaussian(gaussians[i],g);
                        merge = true;
                        break;
                    }
                }

                if(!merge)
                    gaussians.push_back(Gaussian(colMean,rowMean,pixelMean,
                                         colStdDev*stdDevMultiply,rowStdDev*stdDevMultiply,pixelStdDev,ang,N));
            }
        }
    }
//    mergeGaussian(0,1);
}

void Sonar::drawGaussians()
{
    float ri = 0x00, rf = 0xff, dr = rf-ri, r,
          gi = 0x00, gf = 0x00, dg = gf-gi, g,
          bi = 0xff, bf = 0x00, db = bf-bi, b,
          mt=searchThreshold, Mt=0.f, dt, t;

    // Search bigest pixel instensity
    for(unsigned i = 0; i < gaussians.size(); i++)
    {
        if(Mt < gaussians[i].z)
            Mt = gaussians[i].z;
    }

//    Mt = 150;
    cout << "Mt " << Mt << endl;
    dt = Mt-mt;

    for(unsigned i = 0; i < gaussians.size(); i++)
    {
        t = (gaussians[i].z - mt)/dt;
        r = dr*t + ri;
        g = dg*t + gi;
        b = db*t + bi;

//        cout << "color " << t << " ( " << r << " , " << g << " , " << b << ")" << endl;
        ellipse(colorImg,
                Point2f(gaussians[i].x, gaussians[i].y),
                Size2f(gaussians[i].dx, gaussians[i].dy),
                gaussians[i].ang,//180.0*atan2(gaussians[i].dy,gaussians[i].dx)/M_PI,
                0.0 , 360.0,
                Scalar(b,g,r),
                2 , 8 , 0);
    }
}

bool Sonar::intersec(Gaussian &a, Gaussian &b)
{
    float k = mergeStdDevMultiply/stdDevMultiply;
    if( abs(a.x-b.x) <= a.dx*k + b.dx*k &&
        abs(a.y-b.y) <= a.dy*k + b.dy*k)
    {
        return true;
    }
    return false;
}

void Sonar::mergeGaussian(unsigned a, unsigned b)
{
    Gaussian &ga = gaussians[a],
             &gb = gaussians[b];

    ga.dx/= stdDevMultiply; ga.dy /= stdDevMultiply;
    gb.dx/= stdDevMultiply; gb.dy /= stdDevMultiply;

    unsigned N_ab = ga.N + gb.N;
    float xMean_ab = (ga.x * ga.N + gb.x * gb.N)/ ( N_ab ),
            yMean_ab = (ga.y * ga.N + gb.y * gb.y)/ ( N_ab ),
            zMean_ab = (ga.z * ga.N + gb.z * gb.N)/ ( N_ab ),
            xMeanDiff = ga.x-gb.x , yMeanDiff = ga.y - gb.y, zMeanDiff = ga.z - gb.z,
            xStdDev_ab = sqrt(
              ( ( ga.N - 1)*ga.dx*ga.dx + ga.N*ga.x*ga.x + (gb.N-1)*gb.dx*gb.dx + gb.N*gb.x*gb.x
                 -  (ga.N + gb.N) * xMean_ab*xMean_ab ) / (N_ab-1) ),
            yStdDev_ab =   sqrt(abs(
              ( (( ga.N - 1)*ga.dy*ga.dy + ga.N*ga.y*ga.y) + ((gb.N-1)*gb.dy*gb.dy + gb.N*gb.y*gb.y)
                   -  ( (ga.N + gb.N)* yMean_ab*yMean_ab) ) / (N_ab-1) )),
            zStdDev_ab = sqrt(
                ( ( ga.N - 1)*ga.dz*ga.dz + ga.N*ga.z*ga.z + (gb.N-1)*gb.dz*gb.dz + gb.N*gb.z*gb.z
                   -  (ga.N + gb.N) * zMean_ab*zMean_ab ) / (N_ab-1) ),
            ang;

    cout << "====== x ===========" << endl;
    cout << "A = " << (( ga.N - 1)*ga.dx*ga.dx + ga.N*ga.x*ga.x) << endl;
    cout << "B = " << ((gb.N-1)*gb.dx*gb.dx + gb.N*gb.x*gb.x) << endl << endl;

    cout << "A+B = " << (( ga.N - 1)*ga.dx*ga.dx + ga.N*ga.x*ga.x) + ((gb.N-1)*gb.dx*gb.dx + gb.N*gb.x*gb.x) << endl;
    cout << "C = " << ( (ga.N + gb.N)* xMean_ab*xMean_ab) << endl;

    cout << "A+B-C = " << (( ga.N - 1)*ga.dx*ga.dx + ga.N*ga.x*ga.x) + ((gb.N-1)*gb.dx*gb.dx + gb.N*gb.x*gb.x) -  ( (ga.N + gb.N)* xMean_ab*xMean_ab) << endl;

    cout << "====== y ===========" << endl;
    cout << "A = " << (( ga.N - 1)*ga.dy*ga.dy + ga.N*ga.y*ga.y) << endl;
    cout << "B = " << ((gb.N-1)*gb.dy*gb.dy + gb.N*gb.y*gb.y) << endl << endl;

    cout << "A+B = " << (( ga.N - 1)*ga.dy*ga.dy + ga.N*ga.y*ga.y) + ((gb.N-1)*gb.dy*gb.dy + gb.N*gb.y*gb.y) << endl;
    cout << "C = " << ( (ga.N + gb.N)* yMean_ab*yMean_ab) << endl;

    cout << "A+B-C = " << (( ga.N - 1)*ga.dy*ga.dy + ga.N*ga.y*ga.y) + ((gb.N-1)*gb.dy*gb.dy + gb.N*gb.y*gb.y) -  ( (ga.N + gb.N)* yMean_ab*yMean_ab) << endl;

    float Mx, MXy , My , MYx;

    if(ga.x > gb.x)
    {
        Mx = ga.x;     MXy = ga.y;
    }else
    {
        Mx = gb.x;     MXy = gb.y;
    }
    if(ga.y > gb.y)
    {
        My = ga.y;     MYx = ga.x;
    }else
    {
        My = gb.y;     MYx = gb.x;
    }

    ang = calcGaussianAng(xMean_ab, yMean_ab,xStdDev_ab, yStdDev_ab,
                          Mx,MXy,My, MYx);

    cout << "Merge a:" << ga.x << " , " << ga.y << " , " << ga.dx << " , " << ga.dy << endl;
    cout << "Merge b:" << gb.x << " , " << gb.y << " , " << gb.dx << " , " << gb.dy << endl;
    cout << "Resp   :" << xMean_ab << " , " << yMean_ab << " , " << xStdDev_ab << " , " << yStdDev_ab << " , " << N_ab << endl;

    ga.dx*= stdDevMultiply; ga.dy*= stdDevMultiply;
    gb.dx*= stdDevMultiply; gb.dy*= stdDevMultiply;


    gaussians.push_back( Gaussian(xMean_ab, yMean_ab, zMean_ab,
                            xStdDev_ab, yStdDev_ab, zStdDev_ab,
                            ang, N_ab));

/*
    gaussians[a] = Gaussian(xMean_ab, yMean_ab, zMean_ab,
                            xStdDev_ab*stdDevMultiply, yStdDev_ab*stdDevMultiply, zStdDev_ab,
                            ang, N_ab);
    gaussians.erase(gaussians.begin()+b);
 */
}

Gaussian Sonar::pseudoMergeGaussian(Gaussian &ga, Gaussian &gb)
{
    unsigned N_ab = ga.N + gb.N;
    float xMean_ab ,yMean_ab,zMean_ab,
          xMeanDiff = ga.x-gb.x , yMeanDiff = ga.y - gb.y, zMeanDiff = ga.z - gb.z,
          xStdDev_ab,yStdDev_ab,zStdDev_ab;

    xMean_ab = (ga.x + gb.x)/2.f;

    if(ga.dx > gb.dx)
    {
        if(ga.x > gb.x)
            xMean_ab += (ga.dx-gb.dx)/2.f;
        else xMean_ab -= (ga.dx-gb.dx)/2.f;
    }
    else
    {
        if(gb.x > ga.x)
            xMean_ab += (gb.dx-ga.dx)/2.f;
        else xMean_ab -= (gb.dx-ga.dx)/2.f;
    }

    yMean_ab = (ga.y + gb.y)/2.f;

    if(ga.dy > gb.dy)
    {
        if(ga.y > gb.y)
            yMean_ab += (ga.dy-gb.dy)/2.f;
        else yMean_ab -= (ga.dy-gb.dy)/2.f;
    }
    else
    {
        if(gb.y > ga.y)
            yMean_ab += (gb.dy-ga.dy)/2.f;
        else yMean_ab -= (gb.dy-ga.dy)/2.f;
    }

    zMean_ab = (ga.z * ga.N + gb.z * gb.N)/ ( N_ab );
    xMeanDiff = ga.x-gb.x , yMeanDiff = ga.y - gb.y, zMeanDiff = ga.z - gb.z;
    xStdDev_ab = min(ga.dx,gb.dx) + abs(ga.dx-gb.dx)/2.f + abs(xMeanDiff)/2.f;
    yStdDev_ab = min(ga.dy,gb.dy) + abs(ga.dy-gb.dy)/2.f + abs(yMeanDiff)/2.f;
    zStdDev_ab = sqrt(
      ( ( ga.N * (ga.dz*ga.dz) + gb.N * (gb.dz * gb.dz) ) / N_ab  ) +
      ((ga.N * gb.N)/(N_ab*N_ab)) * (zMeanDiff*zMeanDiff) );

    float Mx, MXy , My , MYx,ang;

    if(ga.x > gb.x)
    {
        Mx = ga.x;     MXy = ga.y;
    }else
    {
        Mx = gb.x;     MXy = gb.y;
    }
    if(ga.y > gb.y)
    {
        My = ga.y;     MYx = ga.x;
    }else
    {
        My = gb.y;     MYx = gb.x;
    }

    ang = calcGaussianAng(xMean_ab, yMean_ab,abs(ga.x-gb.x), abs(ga.y-gb.y),
                          Mx,MXy,My, MYx);
    cout << "Merge a:" << ga.x << " , " << ga.y << " , " << ga.dx << " , " << ga.dy << endl;
    cout << "Merge b:" << gb.x << " , " << gb.y << " , " << gb.dx << " , " << gb.dy << endl;
    cout << "Resp   :" << xMean_ab << " , " << yMean_ab << " , " << xStdDev_ab << " , " << yStdDev_ab << endl;

    return Gaussian(xMean_ab, yMean_ab, zMean_ab,
                    xStdDev_ab, yStdDev_ab, zStdDev_ab,
                            ang, N_ab);

/*
    gaussians[a] = Gaussian(xMean_ab, yMean_ab, zMean_ab,
                            xStdDev_ab, yStdDev_ab, zStdDev_ab,
                            ang, N_ab);
    gaussians.erase(gaussians.begin()+b);
*/
}

Sonar::Sonar():
    maxPixel(2000),
    computeMaxPixel(true),
    drawPixelFound(true),
    stdDevMultiply(3),
    mergeStdDevMultiply(4),
    doMerge(true),
    computeGassianAngle(true)
{

}

void Sonar::newImage(Mat img)
{
    Mat img_norm_8bits;
    // Normalize image to convert from 16 bits to 8 bits
    normalize(img,img_norm_8bits, 0 , 255,NORM_MINMAX, CV_8UC1);

    if(computeMaxPixel)
        maxPixel = img.rows * img.cols * 0.001;

    cout << "maxPixel = " << maxPixel << endl;

    // Config histogram
    int grayBins = 256;  // Number of beans
    int histSize[] = { grayBins };

    float grayRange[] = { 0, 256 }; // Range gray scale channel
    const float* histRange[] = { grayRange };

    Mat hist;
    int channels[] = {0}; // Channels that we will process

    calcHist(&img_norm_8bits, 1 , channels , Mat(),
             hist , 1 , histSize , histRange,
             true, false);

    // Analyzing histogram and definig threshold
    unsigned sum = 0;
    for(pixelThreshold = 255 ; pixelThreshold > 0 && sum <= maxPixel; pixelThreshold--)
        sum += hist.at<float>(0,pixelThreshold); // Yes, it's float I don't know why

    cout << " sum = " << sum << " Threshold = " << (unsigned) pixelThreshold << endl;

//    Mat imgBin;
//    threshold(img_norm_8bits, imgBin , pixelThreshold , 255, THRESH_BINARY);

    if(pixelThreshold > 2)
        searchThreshold = pixelThreshold / 2;
    else searchThreshold = pixelThreshold;


    imshow("Norm Image", img_norm_8bits);

    createGaussian(img_norm_8bits);
    drawGaussians();

    imshow("Color Img", colorImg);

}

/*
  Anotacoes:
     Correção de setor:
        - Existem setores com ganho maior que outros talvez por causa da "sujeira" do sensor
     Problema de "borrar" a imagem por causa do mov do robo.
        - Testar calcular média e desvio padrao em funcao de coordenadas polares

     Threshold com decaimento não é bom porque acaba detectando coisas que não deveriam ser detectadas

*/
