#include <cuda_runtime.h>
#include <stdio.h>

__global__ void ballot_sync_test() {
  auto test = 0;
  if (threadIdx.x % 7 == 0) {
    test = 1;
  }
  auto result = __ballot_sync(0xffffffff, test);
  printf("%d\n", result);
  test = 0;
  if (threadIdx.x < 7) {
    test = 1;
  }
  result = __ballot_sync(0xffffffff, test);
  if (threadIdx.x == 0) {
    printf("%d\n", result);
  }
}

int main(int argc, char *argv[]) {
  ballot_sync_test<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}