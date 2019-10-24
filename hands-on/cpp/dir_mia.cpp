#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <iterator>
#include <dirent.h>

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& c)
{
  os << "{ ";
  std::copy(
      std::begin(c),
      std::end(c),
      std::ostream_iterator<T>{os, " "}
  );
  os << '}';

  return os;
}

//std::vector<std::string> entries(/* add the right arguments here */)
//std::vector<std::string> entries(std::shared_ptr<DIR> dir)
//std::vector<std::string> entries(DIR& dir)
std::vector<std::string> entries(DIR* dir)
{
  std::vector<std::string> result;

  // relevant function and data structure are:
  //
  //  int  readdir_r(DIR* dirp, struct dirent* entry, struct dirent** result);
  //  
  //  struct dirent {
  //    // ...
  //    char d_name[256];
  //  };
  
  dirent entry;
  for (auto* r = &entry; readdir_r(dir, &entry, &r) == 0 && r; ) {
    std::cout << "entry.d_name : " << entry.d_name << std::endl;
    result.push_back(entry.d_name);
    // here `entry.d_name` is the name of the current entry
  }

  return result;
}

int main(int argc, char* argv[])
{
  std::string const name = argc > 1 ? argv[1] : ".";
  std::cout << name << '\n';

  // create a smart pointer to a DIR here, with a deleter
  std::shared_ptr<DIR> dir {
    opendir(name.c_str()), 
      [](auto p) {
      std::cout << "closing dir (explicitly)" << std::endl;
      closedir(p);
    } 
  };
  // relevant functions and data structures are
  // DIR* opendir(const char* name);
  // int  closedir(DIR* dirp);

  //  std::vector<std::string> v = entries(/* add the right argument here */);
  std::vector<std::string> v = entries(dir.get());
  std::cout << v << '\n';
}
