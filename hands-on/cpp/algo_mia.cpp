#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c);
std::vector<int> make_vector(int N);

int main()
{
  // create a vector of N elements, generated randomly
  int const N = 10;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // sum all the elements of the vector
  // use std::accumulate
  auto sum = std::accumulate(std::begin(v),std::end(v), 0);
  std::cout << sum << '\n';
  // compute the average of the first half and of the second half of the vector
  auto totsize = v.size();
  std::cout << totsize << '\n';
  int const Nhalf = N/2;
  std::cout << Nhalf << '\n';

  auto test = 2;
  auto sumTEST = std::accumulate(std::begin(v),std::begin(v)+(test),0);
  std::cout << sumTEST << '\n';
  auto aveTEST = sumTEST/float(test);
  std::cout << sumTEST << " " << aveTEST << std::endl;

  auto sum1 = std::accumulate(std::begin(v),std::begin(v)+(Nhalf),0);
  std::cout << sum1 << '\n';
  auto ave1 = sum1/float(Nhalf);
  std::cout << sum1 << " " << ave1 << std::endl;
  std::cout << "ave1: " << ave1 << std::endl;

  auto sum2 = std::accumulate(std::begin(v)+Nhalf,std::end(v),0);
  std::cout << sum2 << '\n';
  auto ave2 = sum2/float(Nhalf);
  std::cout << sum2 << " " << ave2 << std::endl;
  std::cout << "ave2: " << ave2 << std::endl;

  std::cout << sum1+sum2 << '\n';

  // move the three central elements to the beginning of the vector
  // use std::rotate
  std::rotate(std::begin(v),std::begin(v)+Nhalf-1,std::begin(v)+Nhalf+2);
  std::cout << v << std::endl;

  // remove duplicate elements
  // use std::sort followed by std::unique/unique_copy
  std::sort(v.begin(), v.end());
  std::unique(std::begin(v),std::end(v));
  std::cout << v << std::endl;

};

std::ostream& operator<<(std::ostream& os, std::vector<int> const& c)
{
  os << "{ ";
  std::copy(
            std::begin(c),
            std::end(c),
            std::ostream_iterator<int>{os, " "}
            );
  os << '}';

  return os;
}

std::vector<int> make_vector(int N)
{
  std::random_device rd;
  std::mt19937 eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&]() { return dist(eng); });

  return result;
}

