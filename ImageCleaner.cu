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

__device__ int bit_reverse(int n, int numBits) {
  int b = 0;
  for (int i = 0; i < numBits; i++) {
    b = b << 1;
    b = b | (n & 1);
    n = n >> 1;
  }

  return b;
}

__device__ void bit_reverse_copy(float *image, float *image_buf, int stride) {
  int numBits = (SIZEX == 512)? 9 : 10;
  int destinationIndex = bit_reverse(threadIdx.x, numBits);
  image_buf[destinationIndex] = image[threadIdx.x * stride];
}

__device__ void compute_fft_opt(float *input_real, float *input_imag, int size, int stride, int isInverseFFT) {
  int y = threadIdx.x;

  __shared__ float real_image_buf[SIZEX]; // these shared buffers help in reducing the memory latency. Instead of fetching 
  __shared__ float imag_image_buf[SIZEX]; // image pixel data from global memory, the data can now be fetched from shared memory.

  bit_reverse_copy(input_real, real_image_buf, stride);
  bit_reverse_copy(input_imag, imag_image_buf, stride);

  __syncthreads();

  int i = threadIdx.x;
  for (unsigned m = 2; m <= size; m = m*2) {
    int tmp = i / m;
    int k = tmp * m;
    int j = i - k;
    tmp = (j % (m/2));
    float sign = (isInverseFFT == 0) ? -1 : 1;
    float exponent =  (sign) * 2 * PI * tmp / m;
    float fft_real_val = cos(exponent);
    float fft_imag_val = sin(exponent);

    int index1 = (j < m/2)? i : i - (m/2);
    int index2 = (j >= m/2)? i : i + (m/2);

    float real_value = 0;
    float imag_value = 0;
    float ar = real_image_buf[index1];
    float ai = imag_image_buf[index1];
    float br = real_image_buf[index2];
    float bi = imag_image_buf[index2];

    if (j < m/2) {
      real_value = ar + (br * fft_real_val) - (bi * fft_imag_val);
      imag_value = ai + (bi * fft_real_val) + (br * fft_imag_val);
    } else {
      real_value = ar - (br * fft_real_val) + (bi * fft_imag_val);
      imag_value = ai - (bi * fft_real_val) - (br * fft_imag_val);
    }

    __syncthreads();

    real_image_buf[i] = real_value;
    imag_image_buf[i] = imag_value;

    __syncthreads();
  }

  if (isInverseFFT != 0) {
    real_image_buf[i] = real_image_buf[i] / size;
    imag_image_buf[i] = imag_image_buf[i] / size;
  }

  input_real[y * stride] = real_image_buf[y];
  input_imag[y * stride] = imag_image_buf[y];
}

__global__ void cuda_fftx(float *real_image, float *imag_image, int size_x, int size_y)
{
  int x = blockIdx.x; // each row of the image is processed by a different thread block
  compute_fft_opt(&real_image[x*size_x], &imag_image[x*size_x], size_x, 1, 0);
}

__global__ void cuda_ffty(float *real_image, float *imag_image, int size_x, int size_y)
{
  int y = blockIdx.x; // each column is processed by a different thread block.
  compute_fft_opt(&real_image[y], &imag_image[y], size_x, size_x, 0);
}

__global__ void cuda_filter(float *real_image, float *imag_image, int size_x, int size_y)
{
  int eightX = size_x/8;
  int eightY = size_y/8;
  int eight7Y = size_y - eightY;

  int x = blockIdx.x; // each row is processed by a different thread block
  int y = threadIdx.x; // each column of the row is processed by a different thread.

  if(!(x < eightX && y < eightY) &&
        !(x < eightX && y >= eight7Y) &&
        !(x >= eight7Y && y < eightY) &&
        !(x >= eight7Y && y >= eight7Y))
  {
    // Zero out these values
    real_image[y*size_x + x] = 0;
    imag_image[y*size_x + x] = 0;
  }
}

__global__ void cuda_ifftx(float *real_image, float *imag_image, int size_x, int size_y)
{
  int x = blockIdx.x;
  compute_fft_opt(&real_image[x*size_x], &imag_image[x*size_x], size_x, 1, 1);
}

__global__ void cuda_iffty(float *real_image, float *imag_image, int size_x, int size_y)
{
  int y = blockIdx.x;
  compute_fft_opt(&real_image[y], &imag_image[y], size_x, size_x, 1);
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
  int numBlocks = SIZEY;
  int numThreadsPerBlock = SIZEY;

  cuda_fftx<<<numBlocks,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cuda_ffty<<<numBlocks,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cuda_filter<<<numBlocks,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cuda_ifftx<<<numBlocks,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cuda_iffty<<<numBlocks,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);

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

