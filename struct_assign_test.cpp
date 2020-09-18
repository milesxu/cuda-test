#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char *argv[]) {
  auto ptra = make_cudaPitchedPtr(nullptr, 0, 0, 0);
  std::cout << "null ptr: " << ptra.ptr << std::endl;
  auto extent = make_cudaExtent(12 * sizeof(float), 23, 23);
  cudaPitchedPtr ptrb;
  cudaMalloc3D(&ptrb, extent);
  // cudaDeviceSynchronize();
  std::cout << "ptrb: " << ptrb.ptr << " " << ptrb.pitch << " " << ptrb.xsize
            << " " << ptrb.ysize << std::endl;
  ptra = ptrb;
  std::cout << "assigned value: " << ptra.ptr << " " << ptra.pitch << " "
            << ptra.xsize << " " << ptra.ysize << std::endl;
  return 0;
}