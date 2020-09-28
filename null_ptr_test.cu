#include <cuda_runtime.h>
#include <iostream>

__global__ void null_ptr_test(cudaPitchedPtr test) {
  if (threadIdx.x == 0) {
    printf("test args: %ld\n", test.ptr);
  }
}

int main(int argc, char *argv[]) {
  auto ptr = make_cudaPitchedPtr(nullptr, 0, 0, 0);
  null_ptr_test<<<1, 32>>>(ptr);
  //   cudaDeviceSynchronize();
  cudaStreamSynchronize(0);
  std::cout << "result state: " << cudaGetLastError() << std::endl;
  return 0;
}