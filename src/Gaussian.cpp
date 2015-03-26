#include "Gaussian.h"

ostream & operator << (ostream &os , const Gaussian & g )
{
    return os << "( " << g.x << " , " << g.y << " , " << g.z <<
              " ) , ( " << g.dx << " , " << g.dy << " , " << g.dz  <<
              " ) , " << g.ang << " , " << g.N;
}
