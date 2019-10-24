#include <iostream>
#include <chrono>
#include <cassert>
#include <utility>
#include <cstdlib>

using Duration = std::chrono::duration<float>;

// compile time
template<int n>
constexpr auto Pi()
//std::pair<double, Duration> pi(int n)
{
  assert(n > 0);

  //  auto const start = std::chrono::high_resolution_clock::now();

  auto const step = 1. / n;
  auto sum = 0.;
  for (int i = 0; i != n; ++i) {
    auto x = (i + 0.5) * step;
    sum += 4. / (1. + x * x);
  }

  //  auto const end = std::chrono::high_resolution_clock::now();

  return step * sum;
}

std::pair<double, Duration> pi(int n)
{
  assert(n > 0);

  auto const start = std::chrono::high_resolution_clock::now();

  auto const step = 1. / n;
  auto sum = 0.;
  for (int i = 0; i != n; ++i) {
    auto x = (i + 0.5) * step;
    sum += 4. / (1. + x * x);
  }

  auto const end = std::chrono::high_resolution_clock::now();

  return { step * sum, end - start };
}

int main(int argc, char* argv[])
{
  int const n = (argc > 1) ? std::atoi(argv[1]) : 10;
  //  constexpr auto myPi = Pi<n>();
  constexpr int N = 10000; // w/ 0 assert at compile time

  {
    auto const start = std::chrono::high_resolution_clock::now();
    constexpr auto myPiN = Pi<N>();
    auto const end = std::chrono::high_resolution_clock::now();
    std::cout << "myPiN: " << myPiN << " " << Duration{end - start}.count() << "s" << std::endl;
  }

  {
    auto const start = std::chrono::high_resolution_clock::now();
    constexpr auto myPi10 = Pi<10>();
    auto const end = std::chrono::high_resolution_clock::now();
    std::cout << "myPi10: " << myPi10 << " " << Duration{end - start}.count() << "s" << std::endl;
  }

  {
    auto const start = std::chrono::high_resolution_clock::now();
    constexpr auto myPi100 = Pi<100>();
    auto const end = std::chrono::high_resolution_clock::now();
    std::cout << "myPi100: " << myPi100 << " " << Duration{end - start}.count() << "s" << std::endl;
  }

  auto const [value, time] = pi(n);

  std::cout << "pi = " << value
            << " for " << n << " iterations"
            << " in " << time.count() << " s\n";
}
