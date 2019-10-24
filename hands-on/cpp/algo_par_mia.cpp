#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <execution>
#include <chrono>

int main()
{
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::mt19937 eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> uniform_dist{1, MAX_N};

  // fill a vector with SIZE random numbers
  int const SIZE = 100000000;
  std::vector<int> v;
  v.reserve(SIZE);
  std::generate_n(std::back_inserter(v), SIZE, [&]() { return uniform_dist(eng); });

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::accumulate
    auto sum = std::accumulate(std::begin(v),std::end(v),0);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "accumulate std" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::accumulate
    auto sum = std::accumulate(v.begin(),v.end(),0);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "accumulate" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::accumulate
    auto sum = std::accumulate(std::begin(v),std::end(v),0);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "accumulate std" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::reduce, sequential policy
    auto sum = std::reduce(std::execution::seq,std::begin(v),std::end(v),0);
    // NB you need to pass the initial value
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "reduce seq" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::reduce, sequential policy
    auto sum = std::reduce(std::execution::seq,std::begin(v),std::end(v),0);
    // NB you need to pass the initial value
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << sum << " \n";
    std::cout << "reduce (cout) seq" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::reduce, parallel policy
    auto sum = std::reduce(std::execution::par,std::begin(v),std::end(v),0);    
    // NB you need to pass the initial value
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "reduce par" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::reduce, parallel policy
    auto sum = std::reduce(std::execution::par,std::begin(v),std::end(v),0);    
    // NB you need to pass the initial value
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << sum << " \n";
    std::cout << "reduce (cout) par" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort    
    std::sort(std::execution::seq, v.begin(), v.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "sort seq" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort    
    std::sort(v.begin(), v.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "sort (default)" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort    
    std::sort(std::execution::seq, v.begin(), v.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "sort seq" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort, parallel policy
    std::sort(std::execution::par, v.begin(), v.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << "sort par" << " s\n";
    std::cout << " in " << d.count() << " s\n";
  }
}

