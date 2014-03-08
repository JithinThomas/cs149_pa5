#include "ImageCleaner.h"

#ifndef SIZEX
#error Please define SIZEX.
#endif
#ifndef SIZEY
#error Please define SIZEY.
#endif

#define PI  3.14159256

//----------------------------------------------------------------
// TODO:  CREATE NEW KERNELS HERE.  YOU CAN PLACE YOUR CALLS TO
//        THEM IN THE INDICATED SECTION INSIDE THE 'filterImage'
//        FUNCTION.
//
// BEGIN ADD KERNEL DEFINITIONS
//----------------------------------------------------------------

__global__ void cuda_fftx(float *real_image, float *imag_image, int size_x, int size_y)
{
  if (threadIdx.x == 0) {
    printf ("Calling cuda_fftx. size_y = %d", size_y);
  }

  for(unsigned int x = 0; x < size_x; x++)
  {
    //int y = blockDim.x*blockIdx.x + threadIdx.x;
    int y = threadIdx.x;
    // Compute the local values
    float *fft_real = new float[size_y];
    float *fft_imag = new float[size_y];

    for(unsigned int n = 0; n < size_y; n++)
    {
    	float term = -2 * PI * y * n / size_y;
    	fft_real[n] = cos(term);
    	fft_imag[n] = sin(term);
    }

    // Compute the value for this index
    float real_value = 0;
    float imag_value = 0;
    for(unsigned int n = 0; n < size_y; n++)
    {
    	real_value += (real_image[x*size_x + n] * fft_real[n]) - (imag_image[x*size_x + n] * fft_imag[n]);
    	imag_value += (imag_image[x*size_x + n] * fft_real[n]) + (real_image[x*size_x + n] * fft_imag[n]);
    }

    // Reclaim memory
    delete [] fft_real;
    delete [] fft_imag;

    __syncthreads();
    
	  atomicExch(&real_image[x*size_x + y], real_value);
	  atomicExch(&imag_image[x*size_x + y], imag_value);
  }
}

__global__ void cuda_ffty(float *real_image, float *imag_image, int size_x, int size_y)
{
  if (threadIdx.x == 0) {
    printf ("Calling cuda_ffty");
  }

  for(unsigned int y = 0; y < size_y; y++)
  {
    int x = threadIdx.x;
    // Compute the local values
    float *fft_real = new float[size_x];
    float *fft_imag = new float[size_x];
    for(unsigned int n = 0; n < size_y; n++)
    {
      float term = -2 * PI * x * n / size_x;
      fft_real[n] = cos(term);
      fft_imag[n] = sin(term);
    }

    // Compute the value for this index
    float real_value = 0;
    float imag_value = 0;
    for(unsigned int n = 0; n < size_x; n++)
    {
      real_value += (real_image[n*size_x + y] * fft_real[n]) - (imag_image[n*size_x + y] * fft_imag[n]);
      imag_value += (imag_image[n*size_x + y] * fft_real[n]) + (real_image[n*size_x + y] * fft_imag[n]);
    }

    // Reclaim memory
    delete [] fft_real;
    delete [] fft_imag;

    __syncthreads();

    atomicExch(&real_image[x*size_x + y], real_value);
    atomicExch(&imag_image[x*size_x + y], imag_value);
  }
}

__global__ void cuda_filter(float *real_image, float *imag_image, int size_x, int size_y)
{
  if (threadIdx.x == 0) {
    printf ("Calling cuda_filter");
  }

  int eightX = size_x/8;
  //int eight7X = size_x - eightX;
  int eightY = size_y/8;
  int eight7Y = size_y - eightY;
  int x = threadIdx.x;
  for(unsigned int y = 0; y < size_y; y++)
  {
    //int x = threadIdx.x;
    if(!(x < eightX && y < eightY) &&
          !(x < eightX && y >= eight7Y) &&
          !(x >= eight7Y && y < eightY) &&
          !(x >= eight7Y && y >= eight7Y))
    {
      // Zero out these values
      atomicExch(&real_image[y*size_x + x], 0);
      atomicExch(&imag_image[y*size_x + x], 0);
    }
  }
}

__global__ void cuda_ifftx(float *real_image, float *imag_image, int size_x, int size_y)
{
  if (threadIdx.x == 0) {
    printf ("Calling cuda_ifftx");
  }

  for(unsigned int x = 0; x < size_x; x++)
  {
    int y = threadIdx.x;
    // Compute the local values
    float *fft_real = new float[size_y];
    float *fft_imag = new float[size_y];
    for(unsigned int n = 0; n < size_y; n++)
    {
      // Note that the negative sign goes away for the term here
      float term = 2 * PI * y * n / size_y;
      fft_real[n] = cos(term);
      fft_imag[n] = sin(term);
    }

    // Compute the value for this index
    float real_value = 0;
    float imag_value = 0;
    for(unsigned int n = 0; n < size_y; n++)
    {
      real_value += (real_image[x*size_x + n] * fft_real[n]) - (imag_image[x*size_x + n] * fft_imag[n]);
      imag_value += (imag_image[x*size_x + n] * fft_real[n]) + (real_image[x*size_x + n] * fft_imag[n]);
    }

    // Reclaim memory
    delete [] fft_real;
    delete [] fft_imag;

    __syncthreads();

    atomicExch(&real_image[x*size_x + y], real_value/size_y);
    atomicExch(&imag_image[x*size_x + y], imag_value/size_y);
  }
}

__global__ void cuda_iffty(float *real_image, float *imag_image, int size_x, int size_y)
{
  if (threadIdx.x == 0) {
    printf ("Calling cuda_iffty");
  }

  for(unsigned int y = 0; y < size_y; y++)
  {
    int x = threadIdx.x;
    // Compute the local values
    float *fft_real = new float[size_x];
    float *fft_imag = new float[size_x];
    for(unsigned int n = 0; n < size_y; n++)
    {
      // Note that the negative sign goes away for the term
      float term = 2 * PI * x * n / size_x;
      fft_real[n] = cos(term);
      fft_imag[n] = sin(term);
    }

    // Compute the value for this index
    float real_value = 0;
    float imag_value = 0;
    for(unsigned int n = 0; n < size_x; n++)
    {
      real_value += (real_image[n*size_x + y] * fft_real[n]) - (imag_image[n*size_x + y] * fft_imag[n]);
      imag_value += (imag_image[n*size_x + y] * fft_real[n]) + (real_image[n*size_x + y] * fft_imag[n]);
    }

    // Reclaim memory
    delete [] fft_real;
    delete [] fft_imag;

    __syncthreads();

    atomicExch(&real_image[x*size_x + y], real_value/size_x);
    atomicExch(&imag_image[x*size_x + y], imag_value/size_x);
  }
}

//----------------------------------------------------------------
// END ADD KERNEL DEFINTIONS
//----------------------------------------------------------------


__host__ float filterImage(float *real_image, float *imag_image, int size_x, int size_y)
{
  // check that the sizes match up
  assert(size_x == SIZEX);
  assert(size_y == SIZEY);

  int matSize = size_x * size_y * sizeof(float);

  // These variables are for timing purposes
  float transferDown = 0, transferUp = 0, execution = 0;
  cudaEvent_t start,stop;

  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  // Create a stream and initialize it
  cudaStream_t filterStream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream));

  // Alloc space on the device
  float *device_real, *device_imag;
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_real, matSize));
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_imag, matSize));

  // Start timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
  
  // Here is where we copy matrices down to the device 
  CUDA_ERROR_CHECK(cudaMemcpy(device_real,real_image,matSize,cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy(device_imag,imag_image,matSize,cudaMemcpyHostToDevice));
  
  // Stop timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferDown,start,stop));

  // Start timing for the execution
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  //----------------------------------------------------------------
  // TODO: YOU SHOULD PLACE ALL YOUR KERNEL EXECUTIONS
  //        HERE BETWEEN THE CALLS FOR STARTING AND
  //        FINISHING TIMING FOR THE EXECUTION PHASE
  // BEGIN ADD KERNEL CALLS
  //----------------------------------------------------------------

  // This is an example kernel call, you should feel free to create
  // as many kernel calls as you feel are needed for your program
  // Each of the parameters are as follows:
  //    1. Number of thread blocks, can be either int or dim3 (see CUDA manual)
  //    2. Number of threads per thread block, can be either int or dim3 (see CUDA manual)
  //    3. Always should be '0' unless you read the CUDA manual and learn about dynamically allocating shared memory
  //    4. Stream to execute kernel on, should always be 'filterStream'
  //
  // Also note that you pass the pointers to the device memory to the kernel call
  //exampleKernel<<<1,128,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  int numThreadsPerBlock = SIZEY;

  cuda_fftx<<<1,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cuda_ffty<<<1,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cuda_filter<<<1,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cuda_ifftx<<<1,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cuda_iffty<<<1,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);

  //---------------------------------------------------------------- 
  // END ADD KERNEL CALLS
  //----------------------------------------------------------------

  // Finish timimg for the execution 
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&execution,start,stop));

  // Start timing for the transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  // Here is where we copy matrices back from the device 
  CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

  // Finish timing for transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferUp,start,stop));

  // Synchronize the stream
  CUDA_ERROR_CHECK(cudaStreamSynchronize(filterStream));
  // Destroy the stream
  CUDA_ERROR_CHECK(cudaStreamDestroy(filterStream));
  // Destroy the events
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));

  // Free the memory
  CUDA_ERROR_CHECK(cudaFree(device_real));
  CUDA_ERROR_CHECK(cudaFree(device_imag));

  // Dump some usage statistics
  printf("CUDA IMPLEMENTATION STATISTICS:\n");
  printf("  Host to Device Transfer Time: %f ms\n", transferDown);
  printf("  Kernel(s) Execution Time: %f ms\n", execution);
  printf("  Device to Host Transfer Time: %f ms\n", transferUp);
  float totalTime = transferDown + execution + transferUp;
  printf("  Total CUDA Execution Time: %f ms\n\n", totalTime);
  // Return the total time to transfer and execute
  return totalTime;
}

