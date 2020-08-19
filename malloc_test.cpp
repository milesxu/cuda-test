#include <cuda_runtime.h>
#include <iostream>
#include <tuple>

int main(int argc, char *argv[]) {
  auto [w, h, d] = std::tuple(2, 2, 2);
  const auto total_length = w * h * d;
  auto extent = make_cudaExtent(w * sizeof(float), h, d);
  std::cout << extent.width << " " << w * sizeof(float) << std::endl;
  cudaPitchedPtr pitched_ptr;
  cudaMalloc3D(&pitched_ptr, extent);
  const auto unit_length = pitched_ptr.pitch;
  std::cout << unit_length << " " << pitched_ptr.xsize << " "
            << pitched_ptr.ysize << std::endl;
  cudaFree(pitched_ptr.ptr);
  const auto pitched_length = unit_length * h * d;
  const auto more_length = pitched_length + unit_length;
  int *h_data = new int[more_length];
  auto host_ptr = make_cudaPitchedPtr(h_data, w * sizeof(int), w, h);
  std::cout << "host pitch: " << host_ptr.pitch << std::endl;
  int *d_data;
  cudaMalloc(&d_data, more_length);
  cudaMemset(d_data, 0, more_length);
  auto d_ptr = make_cudaPitchedPtr(d_data, unit_length, w, h);
  cudaMemset3D(d_ptr, 1, make_cudaExtent(w * sizeof(int), h, d));
  cudaMemcpy(h_data, d_data, more_length, cudaMemcpyDefault);
  const auto length = more_length / sizeof(int);
  int n = 0, stride = 16;
  for (auto i = 0; i < length; ++i) {
    std::cout << h_data[i] << " ";
    ++n;
    if (n == stride) {
      n = 0;
      std::cout << std::endl;
    }
  }
  delete[] h_data;
  cudaFree(d_data);
  auto null_ptr_test = make_cudaPitchedPtr(nullptr, 0, 0, 0);
  std::cout << null_ptr_test.pitch << " " << null_ptr_test.xsize << " "
            << null_ptr_test.ysize << " " << (null_ptr_test.ptr == nullptr)
            << std::endl;
  return 0;
}