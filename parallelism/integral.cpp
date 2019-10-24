#include <iostream>
#include <iomanip>
#include <mutex>

int main(int argc, char* argv[])
{

  auto f = [](auto x){ 
    unsigned long long local_sum = 0;
    local_sum += 4.0/(1.0+x*x); 
  };

  constexpr unsigned long long num_steps = 1<<20;
  double pi = 0.;
  constexpr double step = 1.0/(double) num_steps;
  double sum = 0.;
  for (unsigned int i=0; i< num_steps; i++){
    auto x = (i+0.5)*step;
    f(x);
    sum = sum + 4.0/(1.0+x*x);
  }
  pi = step * sum;
  std::cout << "result: " << std::setprecision (15) << pi << std::endl;

  return 0;
}
