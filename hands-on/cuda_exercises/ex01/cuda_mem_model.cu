// Includes, System
#include <iostream>
#include <assert.h>
#include <chrono>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0 // 24%8=0

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    cudaSetDevice(MYDEVICE);
    // pointer and dimension for host memory
    int const dimA = (argc > 1) ? std::atoi(argv[1]) : 8;
    std::cout << "dimA " << dimA << std::endl;
    float *h_a;
    //    double *h_a;

    // pointers for device memory
    float *d_a, *d_b;
    //    double *d_a, *d_b;

    // allocate and initialize host memory
    // Bonus: try using cudaMallocHost in place of malloc
    // it has the same syntax as cudaMalloc, but it enables asynchronous copies
    //    h_a = (float *) malloc(dimA*sizeof(float));
    //    size_t memSize = dimA*sizeof(float);
    size_t memSize = dimA*sizeof(*h_a);
    cudaMallocHost( &h_a,memSize );
    for (int i = 0; i<dimA; ++i)
      h_a[i] = i;

    // Part 1 of 5: allocate device memory
    //    size_t memSize = dimA*sizeof(float);
    cudaMalloc( &d_a,memSize );
    cudaMalloc( &d_b,memSize );

    // Part 2 of 5: host to device memory copy
    {
      auto start = std::chrono::system_clock::now();
      cudaMemcpy( d_a, h_a, memSize, cudaMemcpyHostToDevice );
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> dur= stop - start;
      float band = memSize/8./dur.count();
      printf ("h2d : PCI bandwidth %.1f GB/seconds \n",float(memSize)/8000000000./dur.count());
    }

    // Part 3 of 5: device to device memory copy
    {
      auto start = std::chrono::system_clock::now();
      cudaMemcpy( d_b, d_a, memSize, cudaMemcpyDeviceToDevice );
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> dur= stop - start;
      printf ("d2d : PCI bandwidth %.1f GB/seconds \n",float(memSize)/8000000000./dur.count());
    }


    // clear host memory
    for (int i=0; i<dimA; ++i )
        h_a[i] = 0.f;

    // Part 4 of 5: device to host copy
    {
      auto start = std::chrono::system_clock::now();
      cudaMemcpy( h_a, d_a, memSize, cudaMemcpyDeviceToHost );
      auto stop = std::chrono::system_clock::now();
      std::chrono::duration<double> dur= stop - start;
      printf ("d2h : PCI bandwidth %.1f GB/seconds \n",float(memSize)/8000000000./dur.count());
    }


    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy calls");

    // verify the data on the host is correct
    for (int i=0; i<dimA; ++i)
        assert(h_a[i] == (float) i);

    // Part 5 of 5: free device memory pointers d_a and d_b
    cudaFree( d_a );
    cudaFree( d_b );

    // Check for any CUDA errors
    checkCUDAError("cudaFree");

    // free host memory pointer h_a
    //    free(h_a);
    cudaFreeHost(h_a);

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors.  Good work!
    std::cout << "Correct!" << std::endl;

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
