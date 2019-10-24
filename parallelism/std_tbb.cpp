#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>

#include <iostream>
//#include <stdio.h>
//#include <chrono>

int main()
{
  
  // analogous to hardware_concurrency, number of hw threads:
  int n = tbb::task_scheduler_init::default_num_threads();
  std::cout << "n: " << n << std::endl;

  tbb::task_scheduler_init init;
  std::cout << "hello world" << std::endl;

  // or if you wish to force a number of threads:
  int p = 10; //running with 10 threads
  tbb::task_scheduler_init initp(p);

  auto N = 10;
  tbb::parallel_for(
		    tbb::blocked_range<int>(0,N,<G>),
		    [&](const tbb::blocked_range<int>& range)
		    {
		      for(int i = range.begin(); i< range.end(); ++i)
			{
			  x[i]++; }
		    }, <partitioner>);

  return 0;
}
