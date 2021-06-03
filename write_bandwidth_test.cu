#include <iostream>

const auto loop_time = 5000000;
const auto buffer_length = 128;

__global__ void singleThreadWriteShared() {
  __shared__ float buffer[buffer_length];
  if (threadIdx.x == 0) {
    for (auto lp = 0; lp < loop_time; lp++)
      for (auto bi = 0; bi < buffer_length; bi++) {
        const auto tmp = bi * bi;
        buffer[bi] = tmp;
      }
  }
}

__global__ void multiThreadWriteShared() {
  __shared__ float buffer[buffer_length];
  for (auto lp = 0; lp < loop_time; lp++) {
    const auto tmp = threadIdx.x * blockDim.x * 1.0f;
    buffer[threadIdx.x] = tmp;
  }
}

__global__ void singleThreadWriteGlobalNoSync() {
  if (threadIdx.x == 0) {
    auto buffer = (float*)malloc(buffer_length * sizeof(float));
    for (auto lp = 0; lp < loop_time; lp++)
      for (auto bi = 0; bi < buffer_length; bi++) {
        const auto tmp = bi * bi;
        buffer[bi] = tmp;
      }
    free(buffer);
  }
  __syncthreads();
}

__global__ void singleThreadWriteGlobal() {
  __shared__ float* buffer;
  if (threadIdx.x == 0) buffer = (float*)malloc(buffer_length * sizeof(float));
  __syncthreads();
  if (threadIdx.x == 0) {
    for (auto lp = 0; lp < loop_time; lp++)
      for (auto bi = 0; bi < buffer_length; bi++) {
        const auto tmp = bi * bi;
        buffer[bi] = tmp;
      }
  }
  __syncthreads();
  if (threadIdx.x == 0) free(buffer);
}

__global__ void multiThreadWriteGlobal() {
  __shared__ float* buffer;
  if (threadIdx.x == 0) buffer = (float*)malloc(buffer_length * sizeof(float));
  __syncthreads();
  for (auto lp = 0; lp < loop_time; lp++) {
    const auto tmp = threadIdx.x * threadIdx.x * 1.0f;
    buffer[threadIdx.x] = tmp;
  }
  __syncthreads();
  if (threadIdx.x == 0) free(buffer);
}

int main(int argc, char* argv[]) {
  cudaEvent_t start, stop;
  float time, time_m;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  singleThreadWriteShared<<<1, 32>>>();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "Elapsed time: " << time << std::endl;

  cudaEventRecord(start, 0);
  singleThreadWriteShared<<<1, 32>>>();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "Elapsed time again: " << time << std::endl;

  cudaEventRecord(start, 0);
  multiThreadWriteShared<<<1, 128>>>();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "Elapsed time again: " << time << std::endl;

  cudaEventRecord(start, 0);
  singleThreadWriteGlobalNoSync<<<1, 32>>>();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "Single thread global without sync Elapsed time: " << time << " "
            << cudaGetLastError() << std::endl;

  cudaEventRecord(start, 0);
  singleThreadWriteGlobal<<<1, 32>>>();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "Single thread global Elapsed time: " << time << " "
            << cudaGetLastError() << std::endl;

  cudaEventRecord(start, 0);
  multiThreadWriteGlobal<<<1, 128>>>();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_m, start, stop);
  std::cout << "Multiple threads global Elapsed time: " << time_m << " "
            << cudaGetLastError() << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
