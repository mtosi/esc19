---
title: Parallel C++ and TBB
layout: main
section: parallelism
---
### Environment
Make sure you are using the `gcc 9.2.0`:
~~~
$ module load compilers/gcc-9.2.0_sl7
$ gcc --version
gcc (GCC) 9.2.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
~~~

### Hello World
~~~
#include <thread>
#include <iostream>
int main()
{
   auto f = [](int i){
   std::cout << "hello world from thread " << i << std::endl;
  };
//Construct a thread which runs the function f
  std::thread t0(f,0);

//and then destroy it by joining it
  t0.join();
}
~~~

Compile with:
~~~
g++ std_threads.cpp -lpthread -o std_threads
~~~


### Measuring time intervals
~~~
#include <chrono>
...
auto start = std::chrono::system_clock::now();
  foo();
auto stop = std::chrono::system_clock::now();
std::chrono::duration<double> dur= stop - start;
std::cout << dur.count() << " seconds" << std::endl;
~~~
### Exercise 1. Reduction

~~~
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <chrono>

int main(){

  constexpr unsigned int numElements= 100000000;   

  std::vector<int> input;
  input.reserve(numElements);

  std::mt19937 engine;
  std::uniform_int_distribution<> uniformDist(-5,5);
  for ( unsigned int i=0 ; i< numElements ; ++i) input.emplace_back(uniformDist(engine));

  long long int sum= 0;

  auto f= [&](unsigned long long firstIndex, unsigned long long lastIndex){
    for (auto it= firstIndex; it < lastIndex; ++it){
        sum+= input[it];
    }
  };

  auto start = std::chrono::system_clock::now();
  f(0,numElements);
  std::chrono::duration<double> dur= std::chrono::system_clock::now() - start;
  std::cout << "Time spent in reduction: " << dur.count() << " seconds" << std::endl;
  std::cout << "Sum result: " << sum << std::endl;
  return 0;
}
~~~

### Quickly create threads
~~~
unsigned int n = std::thread::hardware_concurrency();
std::vector<std::thread> v;
for (int i = 0; i < n; ++i) {
     v.emplace_back(f,i);
}
for (auto& t : v) {
    t.join();
}
~~~

### Exercise 2. Numerical Integration
~~~
#include <iostream>
#include <iomanip>
#include <chrono>

int main()
{
  double sum = 0.;
  constexpr unsigned int num_steps = 1 << 22;
  double pi = 0.0;
  constexpr double step = 1.0/(double) num_steps;
  auto start = std::chrono::system_clock::now();
  for (int i=0; i< num_steps; i++){
    auto  x = (i+0.5)/num_steps;
    sum = sum + 4.0/(1.0+x*x);
  }
  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> dur= stop - start;
  std::cout << dur.count() << " seconds" << std::endl;
  pi = step * sum;

  std::cout << "result: " <<  std::setprecision (15) << pi << std::endl;
}

~~~




### Exercise 3. pi with Montecarlo

![](montecarlo_pi.png){:height="400px" }.

The area of the circle is pi and the area of the square is 4.

Generate `N` random floats `x` and `y` between `-1` and `1` [https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution](https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution).

Calculate the distance `r` of your point from the origin.

If `r < 1`: the point is inside the circle and increase `Nin`.

The ratio between `Nin` and `N` converges to the ratio between the areas.

To generate random numbers have a look at the following example:
```
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

int main()
{
    // Seed with a real random value, if available
    std::random_device r;

    // Choose a random mean between 1 and 6
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> uniform_dist(1, 6);
    int mean = uniform_dist(e1);
    std::cout << "Randomly-chosen mean: " << mean << '\n';

    // Generate a normal distribution around that mean
    std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 e2(seed2);
    std::normal_distribution<> normal_dist(mean, 2);

    std::map<int, int> hist;
    for (int n = 0; n < 10000; ++n) {
        ++hist[std::round(normal_dist(e2))];
    }
    std::cout << "Normal distribution around " << mean << ":\n";
    for (auto p : hist) {
        std::cout << std::fixed << std::setprecision(1) << std::setw(2)
                  << p.first << ' ' << std::string(p.second/200, '*') << '\n';
    }
}

```
#### Parallel Algos

```
#include <vector>
#include <algorithm>
#include <execution>

int main() {
  constexpr int N = 10000;
  std::vector<int> input;
  std::vector<int> output;
  output.resize(N);
  // fill the vector
  for (int i = 0; i < N; ++i) {
   ....
  }
  // sort it in parallel
  std::sort(std::execution::par, v.begin(), v.end());

  auto foo = [](int i) -> int { return i * -2; };

  // apply a function foo to each element of v 
  std::transform(std::execution::par_unseq, v.begin(), v.end(), output.begin(), foo);
}
```

### Setting the environment for Intel TBB

To enable the use of TBB:

    [student@esc cpp]$ source \
    /storage/gpfs_maestro_home/hpc_software/tbb2019_20191006oss/bin/tbbvars.sh \
    intel64 linux auto_tbbroot

To compile and link:

     [student@esc cpp]$ g++ -O3 algo_par.cpp -std=c++17 -I${TBBROOT}/include -L${TBBROOT}/lib/intel64/gcc4.8 -ltbb

Let's check that you can compile a simple tbb program:
~~~
#include <tbb/tbb.h>
#include "tbb/task_scheduler_init.h"
#include <iostream>
int main()
{
  tbb::task_scheduler_init init;
  std::cout << "Hello World!" << std::endl;
}
~~~

Compile with:
~~~
g++ hello_world.cpp -ltbb -L tbb2019_20191006oss/lib/intel64/gcc4.8/
~~~


### Task parallelism

A task is submitted to a task_group as in the following.
The `run` method is asynchronous. In order to be sure that the task has completed, the `wait` method has to be launched.
Alternatively, the `run_and_wait` method can be used.


~~~
#include <tbb/tbb.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/task_group.h"
#include <iostream>

using namespace tbb;

int Fib(int n) {
    if( n<2 ) {
        return n;
    } else {
        int x, y;
        task_group g;
        g.run([&]{x=Fib(n-1);}); // spawn a task
        g.run([&]{y=Fib(n-2);}); // spawn another task
        g.wait();                // wait for both tasks to complete
        return x+y;
    }
}

int main()
{
  int n = tbb::task_scheduler_init::default_num_threads();
  std::cout << Fib(40) << std::endl;
  return 0;
}
~~~

### Bonus: Graph Traversal

Generate a direct acyclic graph represented as a `std::vector<Vertex> graph` of 20 vertices:
~~~
struct Vertex {
  unsigned int N;
  std::vector<int> Neighbors;
}
~~~

If there is a connection from `A` to `B`, the index of the element `B` in `graph` needs to be pushed into `A.Neighbors`.
Make sure that from the first element of `graph` you can visit the entire graph.

Once generated, when you visit a vertex `X` of the graph, you compute `Fib(X.N)`. Generate `Vertex.N` uniformly between 30 and 40.

Remember to keep track of which vertex has already been visited.
