
int* factory();

// "still reachable"
auto g = factory();

int main()
{
  // "definitely lost"
  auto t = factory();
  delete t; // in order to fix the leak
}

int* factory()
{
  return new int;
}
