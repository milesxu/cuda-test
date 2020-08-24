#include <cuda_runtime.h>
#include <stdio.h>

__device__ int dim1[23];
__device__ int dim2[23][15];

__global__ void array_test() {
  if (threadIdx.x == 0) {
    printf("method 1 result: %ld\n", sizeof(dim1) / sizeof(dim1[0]));
    printf("method 2 result: %ld\n", *(&dim1 + 1) - dim1);
    printf("result from 2 dim array: %ld, %ld\n", *(&dim2 + 1) - dim2,
           *(&dim2[0] + 1) - dim2[0]);
  }
}

int main(int argc, char *argv[]) {
  array_test<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}