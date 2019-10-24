#include <iostream>
#include <assert.h>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 0


// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);
// Part 2 of 4: implement the kernel
__global__ void kernel( int *a, int dimx, int dimy ) 
{
  dim3 index;
  index.x = threadIdx.x + blockIdx.x * blockDim.x;
//  printf("index.x %d threadIdx.x %d blockIdx.x %d blockDim.x %d \n",index.x,threadIdx.x,blockIdx.x,blockDim.x);
  index.y = threadIdx.y + blockIdx.y * blockDim.y;
//  printf("index.y %d threadIdx.y %d blockIdx.y %d blockDim.y %d \n",index.y,threadIdx.y,blockIdx.y,blockDim.y);

  if (index.x < dimx && index.y < dimy ) {
    int i = index.x + index.y * dimx;
//    printf("i %d x %d y %d dimx %d \n",i,index.x,index.y,dimx);
    a[i] = i;
  }

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
    cudaSetDevice(MYDEVICE);
// Part 1 and 4 of 4: set the dimensions of the matrix
    int const dimx = (argc > 4) ? std::atoi(argv[1]) : 4;
    int const dimy = (argc > 4) ? std::atoi(argv[2]) : 4;
    std::cout << "MATRIX " << dimx << "x" << dimy << std::endl;
    //    int dimx = 4;
    //    int dimy = 4;
    int num_bytes = dimx*dimy*sizeof(dimx);

    int *d_a=0, *h_a=0; // device and host pointers

    //    h_a = (int*)malloc(num_bytes);
    cudaMallocHost( &h_a,num_bytes );
    //allocate memory on the device
    cudaMalloc( &d_a,num_bytes );

    if( NULL==h_a || NULL==d_a ) {
        std::cerr << "couldn't allocate memory" << std::endl;
        return 1;
    }

    // Part 2 of 4: define grid and block size and launch the kernel
    dim3 grid, block;
    //    block.x = 2;
    //    block.y = 2;
    block.x = (argc > 4) ? std::atoi(argv[3]) : 2;
    block.y = (argc > 4) ? std::atoi(argv[4]) : 2;
    std::cout << "#THREADS " << block.x << "x" << block.y << std::endl;
    grid.x  = (dimx + block.x - 1)/block.x;
    grid.y  = (dimy + block.y - 1)/block.y;
    std::cout << "#BLOCKS " << grid.x << "x" << grid.y << std::endl;
    

    kernel<<<grid, block>>>( d_a, dimx, dimy );
    // block until the device has completed
    cudaDeviceSynchronize();
    // check if kernel execution generated an error
    checkCUDAError("kernel execution");
    // device to host copy
    cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy");
    // verify the data returned to the host is correct
    for(int row=0; row<dimy; row++)
    {
        for(int col=0; col<dimx; col++)
            assert(h_a[row * dimx + col] == row * dimx + col);
    }
    // free host memory
    //    free( h_a );
    cudaFreeHost(h_a);

    // free device memory
    cudaFree( d_a );

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
