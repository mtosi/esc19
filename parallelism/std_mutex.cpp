#include <thread>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <mutex>
#include <vector>
#include <numeric>

int main(int argc, char* argv[])
{
  std::mutex myMutex;
  unsigned long long sum = 0;
  int const N = (argc > 1) ? std::atoi(argv[1]) : 100000; // 10
  std::vector<int> vec(N);
  for (unsigned int i = 0; i < vec.size(); ++i)
    vec[i] = i;


  // construct a thread which runs the function f
  unsigned int n = std::thread::hardware_concurrency();
  n = 4;

  auto chunck = N/n;
  std::cout << "chunck: " << chunck << std::endl;

  auto f = [&](unsigned int my_id) {
    //    std::cout << "hello world from thread " << my_id << std::endl;
    //    std::cout << "hello world" << " " << "from thread " << my_id << std::endl;
    printf ("hello world from thread %d \n",my_id);
    //critical section begins here
    std::cout << "Only one thread at a time" << std::endl;

    auto start = std::chrono::system_clock::now();
    auto i_start = my_id * chunck;
    auto i_end = i_start + chunck;
    if (my_id == (n-1))
      i_end = N;
    unsigned long long local_sum = std::accumulate(std::begin(vec)+i_start,std::begin(vec)+i_end,(unsigned long long)(0)); // !!!! auto makes use of int !!! and it is not enough (10^9)    
    printf ("local_sum:  %lld (%d)\n",local_sum,my_id);
    printf ("sum:  %lld (%d)\n",sum,my_id);

    // lock only when it is needed !!!
    {
      std::lock_guard<std::mutex> myLock(myMutex);
      sum += local_sum;
    }

    printf ("sum:  %lld (%d)\n",sum,my_id);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> dur= stop - start;
    //    std::cout << "from thread " << my_id << " " << dur.count() << " seconds" << std::endl; // move to printf !!!!
    printf ("from thread %d : %f seconds \n",my_id,dur.count());
  };

  std::vector<std::thread> v;
  for (unsigned int i = 0; i < n; ++i) {
    v.emplace_back(f,i);
  }
  std::cout << "sum: " << sum << std::endl;
  // and then destroy it by joining it (synchronization happens --> barrier)
  for (auto& t : v) {
    t.join();
  }
  std::cout << "sum: " << sum << std::endl;
  return 0;
}
