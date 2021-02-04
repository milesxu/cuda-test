#include <cuda_runtime.h>
#include <iostream>

__device__ int test[2];
__constant__ float pi;

int *store[] = {test};

__global__ void assign_test() {
  if (threadIdx.x == 0) {
    test[0] = 23;
    test[1] = 15;
    printf("%f\n", pi);
  }
}

int main(int argc, char *argv[]) {
  std::cout << test << std::endl << store[0] << std::endl;
  const auto h_pi = 2.0f * std::asin(1.0f);
  cudaMemcpyToSymbolAsync(pi, &h_pi, sizeof(float), 0, cudaMemcpyDefault);
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