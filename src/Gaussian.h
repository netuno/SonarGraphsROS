#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include<iostream>
using namespace std;

class Gaussian
{
public:
    Gaussian(){}
    Gaussian(float x, float y , float z,
             float dx, float dy, float dz, float ang, unsigned N):
        x(x),y(y),z(z),dx(dx),dy(dy),dz(dz),ang(ang),N(N){}
    float x, y, z, ang;
    float dx,dy,dz;
    unsigned N;
};

ostream & operator << (ostream &os , const Gaussian & g );

#endif // GAUSSIAN_H
