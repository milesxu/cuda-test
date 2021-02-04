#include <iostream>

class A1 {
public:
  char bb1;
  short bb;
  int a;
  double aa;
};

class A2 {
public:
  double aa;
  char bb1;
  int a;
  short bb;
} __attribute((packed)) __;

class A3 {
public:
  double aa;
  char bb1;
  int a;
};

int main(int, char **) {
  std::cout << sizeof(A1) << " " << sizeof(A2) << " " << sizeof(A3)
            << std::endl;
  return 0;
}