#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void cooperative_group_test() {
  if (threadIdx.x == 0) {
    printf("block Dim: %d\n", blockDim.x);
  }
  auto block = cg::this_thread_block();
  if (block.thread_rank() == 0) {
    printf("block size: %d\n", block.size());
  }
  cg::thread_block_tile<32> tile_warp = cg::tiled_partition<32>(block);
  cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);
  if (tile4.thread_rank() == 0) {
    printf("group size: %d\n", tile4.size());
    // printf("meta group size: %d\n", tile_warp.meta_group_size());
    // printf("meta group rank: %d\n", tile_warp.meta_group_rank());
  }
}

int main(int argc, char *argv[]) {
  const auto blocks = 1;
  const auto threads_per_block = 128;
  printf("begin to call gpu kernels...\n");
  cooperative_group_test<<<blocks, threads_per_block>>>();
  cudaDeviceSynchronize();
  std::cout << cudaGetLastError() << " is the error code." << std::endl;
  return 0;
}