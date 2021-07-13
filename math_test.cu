#include <cuda_runtime.h>
#include <stdio.h>

__global__ void LogTest() {
  printf("the log value of 97.105705f: %f\n", logf(97.105705f));
}

int main(int argc, char *argv[]) {
  LogTest<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}