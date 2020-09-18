#include <cuda_runtime.h>
#include <iostream>

__device__ int test[2];

int *store[] = {test};

__global__ void assign_test() {
  if (threadIdx.x == 0) {
    test[0] = 23;
    test[1] = 15;
  }
}

int main(int argc, char *argv[]) {
  std::cout << test << std::endl << store[0] << std::endl;
  assign_test<<<1, 1>>>();
  int result[2];
  int *d_r;
  cudaGetSymbolAddress((void **)&d_r, test);
  store[0] = d_r;
  //   cudaMemcpyFromSymbolAsync(result, d_r, 2 * sizeof(int), 0,
  //   cudaMemcpyDefault);
  cudaStreamSynchronize(0);
  std::cout << cudaGetLastError() << std::endl;
  cudaMemcpyAsync(result, store[0], 2 * sizeof(int), cudaMemcpyDefault);
  cudaStreamSynchronize(0);
  std::cout << "error: " << cudaGetLastError() << std::endl
            << result[0] << " " << result[1] << std::endl;
}