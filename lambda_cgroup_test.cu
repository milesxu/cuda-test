#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace cg = cooperative_groups;

const int tile_size = 4;

__global__ void LambdaCooperativeGroupTest(int loop_time) {
  const auto output_func = [&](int cg_row, int size, int meta_id) {
    printf("status: thread: %d, group row: %d, group size: %d, meta id: %d\n",
           threadIdx.x, cg_row, size, meta_id);
  };

  const auto block = cg::this_thread_block();
  const auto tile_row = cg::tiled_partition<tile_size>(block);
  const auto cg_row = tile_row.thread_rank();
  for (auto col = tile_row.meta_group_rank(); col < loop_time;
       col += tile_row.meta_group_size()) {
    output_func(cg_row, tile_row.meta_group_size(), tile_row.meta_group_rank());
  }
}

int main(int argc, char *argv[]) {
  LambdaCooperativeGroupTest<<<1, 128>>>(49);
  cudaStreamSynchronize(0);
  return 0;
}