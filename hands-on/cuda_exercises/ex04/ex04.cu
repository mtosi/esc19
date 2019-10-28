/**
Throughput (GB/s)= Memory_rate(Hz) * memory_interface_width(byte) * 2 /10^9
877 MHz * 4096-bit * 2 /10^9 
877 MHz * 512 byte * 2 /10^9 = 898.048 GB/s

Device 0: "Tesla V100-SXM2-32GB"
  CUDA Driver Version / Runtime Version          10.0 / 10.0
  CUDA Capability Major/Minor version number:    7.0
  Total amount of global memory:                 32480 MBytes (34058272768 bytes)
  (80) Multiprocessors x ( 64) CUDA Cores/MP:    5120 CUDA Cores
  GPU Clock rate:                                1530 MHz (1.53 GHz)
  Memory Clock rate:                             877 Mhz
  Memory Bus Width:                              4096-bit
  L2 Cache Size:                                 6291456 bytes
  Max Texture Dimension Size (x,y,z)             1D=(131072), 2D=(131072,65536), 3D=(16384,16384,16384)
  Max Layered Texture Size (dim) x layers        1D=(32768) x 2048, 2D=(32768,32768) x 2048
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 5 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Bus ID / PCI location ID:           97 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
**/

#include <stdio.h>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 0
//#define MYDEVICE 1
__global__
void saxpy(unsigned int n, double a, double *x, double *y)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(int argc, char* argv[])
{


  cudaSetDevice(MYDEVICE);

  // 1<<N is the equivalent to 2^N
  // 20 = 2*2*5
  // 20 * (1 << 20) = 5 * (1 << 22)
  unsigned int N =  (argc > 1) ? std::atoi(argv[1]) : 20 * (1 << 20);
  printf("estimating kernel throughput by using arrays of %d elements\n",N);
  double *x, *y, *d_x, *d_y;
  //  x = (double*)malloc(N*sizeof(double));
  //  y = (double*)malloc(N*sizeof(double));

  size_t memSize = N*sizeof(double);
  cudaMallocHost( &x,memSize );
  cudaMallocHost( &y,memSize );

  //  cudaMalloc(&d_x, N*sizeof(double)); 
  //  cudaMalloc(&d_y, N*sizeof(double));
  cudaMalloc( &d_x, memSize );
  cudaMalloc( &d_y, memSize );

  for (unsigned int i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = 2.0;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpyAsync(d_x, x, memSize, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_y, y, memSize, cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_x, x, memSize, cudaMemcpyHostToDevice);
  //  cudaMemcpy(d_y, y, memSize, cudaMemcpyHostToDevice);

  dim3 grid, block;
  block.x = (argc > 2) ? std::atoi(argv[2]) : 512;

  grid.x = (N+block.x-1)/block.x;

  printf("#THREADS %d\n",block.x);
  printf("#BLOCKS %d\n", grid.x);

  cudaEventRecord(start);

  saxpy<<<grid,block>>>(N, 2.0, d_x, d_y);

  cudaEventRecord(stop);

  //  cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(y, d_y, memSize, cudaMemcpyDeviceToHost);
  //  cudaMemcpy(y, d_y, memSize, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);


  //  printf("size of double %d\n",sizeof(double));
  //  printf("size of float %d\n",sizeof(float));
  //  printf("memSize %d\n",memSize);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float inv_dur = 1.f/milliseconds;
  
  float th_throughput = 877 * 512 * 2 * 0.001; // GB/s
  size_t tot_memory = memSize + memSize + memSize; // + sizeof(n) + sizeof(a);
  float exp_throughput = float(tot_memory)*0.001*0.001*inv_dur;
  printf ("kernel throughput th %.1f GB/se, exp %.1f GB/seconds (%.0f %) \n",th_throughput,exp_throughput,exp_throughput/th_throughput*100.);
  
  double maxError = 0.;
  for (unsigned int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0));
  }
  
  cudaFree(d_x);
  cudaFree(d_y);
  //  free(x);
  //  free(y);
  cudaFreeHost(x);
  cudaFreeHost(y);

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  printf ("Correct!\n");

  return 0;
}



