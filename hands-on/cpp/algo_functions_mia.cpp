#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>

int op_square (int i);
std::ostream& operator<<(std::ostream& os, std::vector<int> const& c);
std::vector<int> make_vector(int N);
bool isMultiple();

int main()
{
  // create a vector of N elements, generated randomly
  int const N = 10;
  //  int const N = 2;
  std::vector<int> v = make_vector(N);
  std::cout << v << '\n';

  // multiply all the elements of the vector
  // use std::accumulate
  auto m = std::accumulate(std::begin(v),std::end(v),1,std::multiplies<>{});
  std::cout << m << std::endl;
  uint64_t m64 = std::accumulate(std::begin(v),std::end(v),1,std::multiplies<>{});
  std::cout << m64 << std::endl;
  unsigned long long mULL = std::accumulate(std::begin(v),std::end(v),1,std::multiplies<>{});
  std::cout << mULL << std::endl;
  // sort the vector in descending order
  // use std::sort
  std::sort(std::begin(v),std::end(v),std::greater<>{});
  std::cout << v << std::endl;

  // move the even numbers at the beginning of the vector
  // use std::partition
  auto it = std::partition(std::begin(v), std::end(v), [](int i){return i % 2 == 0;});
 
  std::cout << "\nPartitioned vector:\n    ";
  std::copy(std::begin(v), it, std::ostream_iterator<int>(std::cout, " "));
  std::cout << " * ";
  std::copy(it, std::end(v), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // create another vector with the squares of the numbers in the first vector
  // use std::transform
  std::vector<int> v_square(v.size());
  //  v_square.resize(v.size());                         // allocate space
  std::transform (v.begin(), v.end(), v_square.begin(), op_square);
  std::cout << v_square << std::endl;

  std::vector<int> v_square_lambda(v.size());
  //  v_square_lambda.resize(v.size());                         // allocate space
  std::transform (v.begin(), v.end(), v_square_lambda.begin(), [](int i) {return i*i;});
  std::cout << v_square_lambda << std::endl;

  // find the first multiple of 3 or 7
  // use std::find_if
  std::cout << std::find_if(v.begin(),v.end(),[](int i){return i=3;}) << std::endl;

  // erase from the vector all the multiples of 3 or 7
  // use std::remove_if followed by vector::erase

};

int op_square (int i) { return i*i; }

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
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::mt19937 eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> dist{1, MAX_N};

  std::vector<int> result;
  result.reserve(N);
  std::generate_n(std::back_inserter(result), N, [&]() { return dist(eng); });

  return result;
}
