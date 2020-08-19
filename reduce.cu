//环境：module load cuda/10.2.89
//编译指令：nvcc --gpu-architecture=sm_70 reduce.cu -o reduce
//运行指令：srun -N 1 --gres=gpu:1 ./reduce
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

const size_t dim = 51200;
const size_t block_num = 160;
const size_t threads_per_block = 512;
const int warp_size = 32;
const int full_mask = 0xffffffff;

__global__ void one_assign(int *array, size_t length)
{
    const size_t global_dim = gridDim.x * blockDim.x;
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t unit_num = length / global_dim;
    const size_t start = unit_num * global_id;
    const size_t end = start + unit_num;
    for (size_t i = start; i < end; i++)
        array[i] = 1;
}

__inline__ __device__ size_t warpReduce(size_t val)
{
    for (unsigned int i = warp_size / 2; i > 0; i /= 2)
    {
        val += __shfl_down_sync(full_mask, val, i);
    }
    return val;
}

__global__ void reduce_to_buffer(int *array, size_t length, size_t *buffer)
{
    const size_t global_dim = gridDim.x * blockDim.x;
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t unit_num = length / global_dim;
    const size_t start = unit_num * global_id;
    const size_t end = start + unit_num;

    size_t thread_sum = 0;
    for (size_t i = start; i < end; i++)
    {
        thread_sum += array[i];
    }

    thread_sum = warpReduce(thread_sum);
    const int lane_id = threadIdx.x & 31;
    const int warp_id = global_id / warp_size;
    if (lane_id == 0)
        buffer[warp_id] = thread_sum;
}

__global__ void reduce_in_buffer(size_t *buffer, size_t buffer_length)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_id / warp_size;
    if (warp_id == 0)
    {
        size_t thread_sum = 0;
        for (size_t i = threadIdx.x; i < buffer_length; i += warp_size)
            thread_sum += buffer[i];
        thread_sum = warpReduce(thread_sum);
        if (global_id == 0)
            buffer[0] = thread_sum;
    }
}

int main(int argc, char *argv[])
{
    cout << "Reduce program begin..." << endl;
    int *matrix;
    cudaMallocManaged(&matrix, dim * dim * sizeof(int), cudaMemAttachGlobal);
    cout << dim * dim << endl;
    one_assign<<<block_num, threads_per_block>>>(matrix, dim * dim);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    size_t *buffer;
    const int buffer_length = block_num * threads_per_block / warp_size;
    cudaMallocManaged(&buffer, buffer_length * sizeof(size_t),
                      cudaMemAttachGlobal);
    reduce_to_buffer<<<block_num, threads_per_block>>>(matrix, dim * dim,
                                                       buffer);
    reduce_in_buffer<<<1, 32>>>(buffer, buffer_length);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << buffer[0] << endl;
    cout << "Totally used time: " << elapsedTime << "ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(matrix);
    cudaFree(buffer);
    cout << "Reduce program end..." << endl;
    return 0;
}