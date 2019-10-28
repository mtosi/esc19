#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>
#include<chrono>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

constexpr bool DEBUG = false; 
constexpr size_t BLOCK_SIZE = 512;
constexpr size_t SHARE_BLOCK_SIZE = 2*512;

double random_double(void)
{
  return static_cast<double>(rand()) / RAND_MAX;
}


// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);


// Part 1 of 6: implement the kernel
__global__ void block_sum(const double *input,
                          double *per_block_results,
			  const size_t n)
{
  
  //fill me
  __shared__ double sdata[SHARE_BLOCK_SIZE];

  int g_index = threadIdx.x + blockIdx.x * blockDim.x;
  int s_index = threadIdx.x;

  sdata[s_index] = (g_index < n) ? input[g_index] : 0.;
  
  // Synchronize (ensure all the data is available)
  __syncthreads();
  
  int sdata_size = SHARE_BLOCK_SIZE;
  bool loop = (n == 1) ? false : true;
  while (__syncthreads_or(loop) ) {
    int r_index = sdata_size-1-s_index;

    // reduction origami  
    if(r_index >= (sdata_size+1)/2 && r_index <= sdata_size) {
      sdata[s_index] += sdata[r_index];
      //      sdata[r_index] = 0;
    }    
    sdata_size = (sdata_size+1) / 2;
    if(sdata_size==1) loop = false;
  }
  if (s_index==0)
    per_block_results[blockIdx.x] = sdata[s_index];

  if (DEBUG) {
    if (blockIdx.x < 20 && (g_index < n) && s_index < 3 )
      printf("per_block_results[%d]] %f %d\n",blockIdx.x,per_block_results[blockIdx.x],s_index);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{

  // create array of 256ki elements
  // 1<<N is the equivalent to 2^N
  const int num_elements = (argc > 1) ? std::atoi(argv[1]) : 1<<18;
  //  bool DEBUG = (num_elements < 1000 ? true : false);
  srand(time(NULL));
  // generate random input on the host
  std::vector<double> h_input(num_elements);
  for(int i = 0; i < h_input.size(); ++i) {
    //    h_input[i] = random_double();
    h_input[i] = double(i);
    if (DEBUG) std::cout << " " << h_input[i];
  }
  if (DEBUG) std::cout << std::endl;
  
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    const double host_result = std::accumulate(h_input.begin(), h_input.end(), 0.);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cerr << "Host sum: " << host_result << " in " << d.count() << " s" << std::endl;
  }
  // WITH ONLY 1 STREAM IS NOT NECESSARY
  //  cudaEvent_t cuda_start, cuda_stop;
  //  cudaEventCreate(&cuda_start);
  //  cudaEventCreate(&cuda_stop);

  {

    auto t0 = std::chrono::high_resolution_clock::now();
    //Part 1 of 6: move input to device memory
    size_t memSize = num_elements*sizeof(double);
    double *d_input = 0;
    cudaMalloc( &d_input,memSize );
    cudaMemcpyAsync( d_input, &h_input[0], memSize, cudaMemcpyHostToDevice );

    auto t = std::chrono::high_resolution_clock::now();
    {
      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> d = t1 - t0;
      std::cout << "h2d cudaMemcpyAsync " << d.count() << " s" << std::endl;
    }

    dim3 num_blocks, block_size;
    block_size.x = BLOCK_SIZE;
    num_blocks.x = ( num_elements + block_size.x - 1 )/block_size.x;
    // Part 2 of 6: allocate the partial sums: How much space does it need?
    double *d_partial_sums_and_total = 0;
    memSize = num_blocks.x*sizeof(double);
    cudaMalloc( &d_partial_sums_and_total,memSize );
    
    t = std::chrono::high_resolution_clock::now();
    {
      //  cudaEventRecord(cuda_start);
      // Part 3 of 6: launch one kernel to compute, per-block, a partial sum. How much shared memory does it need?
      block_sum<<<num_blocks,block_size>>>(d_input, d_partial_sums_and_total, num_elements);
      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> d = t1 - t0;
      std::cout << "block_sum<<<num_blocks,block_size>>> " << d.count() << " s " << std::chrono::duration<float>(t1-t).count() << " s" << std::endl;
    }

    t = std::chrono::high_resolution_clock::now();
    {
      // Part 4 of 6: compute the sum of the partial sums
      block_sum<<<1,block_size>>>(d_partial_sums_and_total, &d_partial_sums_and_total[0], num_blocks.x);
      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> d = t1 - t0;
      std::cout << "block_sum<<<1,block_size>>> " << d.count() << " s " << std::chrono::duration<float>(t1-t).count() << " s" << std::endl;
    }

    // check if kernel execution generated an error
    checkCUDAError("kernel execution");
    
    t = std::chrono::high_resolution_clock::now();
    {
      // Part 5 of 6: copy the result back to the host
      double device_result = 0;
      memSize = sizeof(double);
      cudaMemcpyAsync(&device_result, &d_partial_sums_and_total[0], memSize, cudaMemcpyDeviceToHost);
      
      // Part 6 of 6: deallocate device memory
      cudaFree(d_input);
      cudaFree(d_partial_sums_and_total);
      
      auto t1 =std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> d = t1 - t0;
      //    float milliseconds = 0;
      //  cudaEventElapsedTime(&milliseconds, cuda_start, cuda_stop);
      std::cout << "Device sum: " << device_result << " in " << d.count() << " s " << std::chrono::duration<float>(t1-t).count() << " s" << std::endl;
    }
  }
  return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        std::cerr << "Cuda error: " << msg << " " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }                         
}
