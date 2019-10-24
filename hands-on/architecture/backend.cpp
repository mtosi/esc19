#include <cmath>

struct Point {

   double norm() const { return std::sqrt(x*x+y*y+z*z);}
   
   double x,y,z;
   bool ok;
};

#include<random>

std::mt19937 reng;
std::normal_distribution<double> gauss(10.,1.);

void fill(Point & p) {
  p.x = gauss(reng);
  p.y = gauss(reng);
  p.z = gauss(reng);
}

Point cross(Point const & a, Point const & b) {
  Point c;
  c.x = a.y*b.z+a.z*b.z;
  c.y = a.x*b.z-a.z*b.x;  
  c.z = a.y*b.x+a.x*b.y;
  return c;
}

#include<vector>
#include <chrono>
#include<iostream>



int main() {

 auto start = std::chrono::high_resolution_clock::now();
 auto delta=start-start;

  std::vector<Point> pointsI(1024*8);
  std::vector<Point> pointsO(1024*8);
  for (auto &p : pointsI) fill(p);
  double sum=0;
  for (int k=0; k<10000; ++k) {
    if (k>0) delta -= (std::chrono::high_resolution_clock::now()-start);    
    for (int i=0, n=pointsI.size(); i<n; ++i) {
      /* WRONG !!! 
	 - if pointsO=pointsI it is wrong
	 - makes x3 norm()
	 - makes x3 division
      pointsO[i].x = pointsI[i].x/pointsI[i].norm();
      pointsO[i].y = pointsI[i].y/pointsI[i].norm();
      pointsO[i].z = pointsI[i].z/pointsI[i].norm();
      */
      double inv_norm_pointI = 1./pointsI[i].norm();
      pointsO[i].x = pointsI[i].x*inv_norm_pointI;
      pointsO[i].y = pointsI[i].y*inv_norm_pointI;
      pointsO[i].z = pointsI[i].z*inv_norm_pointI;
    }
    if (k>0) delta += (std::chrono::high_resolution_clock::now()-start);
    for (auto &p : pointsO) sum+=p.x;

  }

  auto deltaF = std::chrono::duration_cast<std::chrono::milliseconds>(delta).count();
  std::cout << "norm time " << deltaF << " " << sum << std::endl;


  delta=start-start;

  std::vector<Point> pointsI2(1024*8);
  for (auto &p : pointsI2) fill(p);
    for (int k=0; k<10000; ++k) {
    if (k>0) delta -= (std::chrono::high_resolution_clock::now()-start);
    for (int i=0, n=pointsI.size(); i<n; ++i) {
      pointsO[i] = cross(pointsI[i],pointsI2[i]);
    }
    if (k>0) delta += (std::chrono::high_resolution_clock::now()-start);
    for (auto &p : pointsO) sum+=p.x;
  } 
  
 deltaF = std::chrono::duration_cast<std::chrono::milliseconds>(delta).count();
 std::cout << "cross time " << deltaF << " " << sum << std::endl;



}
