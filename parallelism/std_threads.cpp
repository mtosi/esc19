#include <thread>
#include <iostream>
#include <stdio.h>
#include <chrono>

int main()
{
  auto f = [](int i) {
    auto start = std::chrono::system_clock::now();
    //    std::cout << "hello world from thread " << i << std::endl;
    std::cout << "hello world" << " " << "from thread " << i << std::endl;
    //    printf ("hello world from thread %d \n",i);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> dur= stop - start;
    //    std::cout << "from thread " << i << " " << dur.count() << " seconds" << std::endl; // move to printf !!!!
    printf ("from thread %d : %f seconds \n",i,dur.count());
  };

  // construct a thread which runs the function f
  std::thread t0(f,0);
  std::thread t1(f,1);
  std::thread t2(f,2);
  
  // and then destroy it by joining it (synchronization happens --> barrier)
  t0.join();
  t1.join();
  t2.join();

  return 0;
}
