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

// Given an integer 'n', reverses its bits and returns the resulting integer
__device__ int bit_reverse(int n, int numBits) {
  int b = 0;
  for (int i = 0; i < numBits; i++) {
    b = b << 1;
    b = b | (n & 1);
    n = n >> 1;
  }

  return b;
}

// Used for optimized FFT. Prior to starting the FFT algorithm, the input array needs to be shuffled and rewritten in a specific order
// For each source index 'i', we compute the destination index 'j' (j == bit_reverse(i)) and then copy A[i] to A[j]
// The function also takes an additional parameter which may be '1' or SIZEX depending on whether fftx or ffty invoked it.
__device__ void bit_reverse_copy(float *src_buf, float *dst_buf, int stride) {
  int numBits = (SIZEX == 512)? 9 : 10;
  int destinationIndex = bit_reverse(threadIdx.x, numBits);
  dst_buf[destinationIndex] = src_buf[threadIdx.x * stride];
}

// Given two arrays (ie, one for real and one for imaginaary values), this function computes the FFT/IFFT for each
// using an iterative version of the Cooley-Tukey algorithm
// Input parameters:
//    * real_image   : pointer to array of real values
//    * imag_image   : pointer to array of imaginary values
//    * size         : length of the input arrays
//    * stride       : stride used to access elements in the input array. In the case of fftx/ifftx, stride == 1 and for ffty/iffty, stride == SIZEX
//    * isInverseFFT : Set to 0 for fftx/ffty and to 1 for iffx/iffty. If enabled, the sign of the exponent is set to '1' and the final value is scaled
//                     down by SIZEX.
//    * enableFilter : This is enabled only for ffty. It writes out 0 to those cells that would eventually get filtered out by cuda_filter.
//                     This is an optimization that avoids the need to invoke cuda_filter after computing ffty
__device__ void compute_fft(float *real_image, float *imag_image, int size, int stride, int isInverseFFT, int enableFilter) {
  int i = threadIdx.x;

  __shared__ float real_image_buf[SIZEX]; // these shared buffers help in reducing the memory latency. Instead of fetching 
  __shared__ float imag_image_buf[SIZEX]; // image pixel data from global memory, the data can now be fetched from shared memory.

  // Shuffle (using bit-reverse algorithm) and copy the input arrays to shared memory
  bit_reverse_copy(real_image, real_image_buf, stride);
  bit_reverse_copy(imag_image, imag_image_buf, stride);

  // Shared variable to store precomputed sine/cosine values. This helps a lot since a lot of the cosine/sine computations are repeated.
  __shared__ float cos_arr[SIZEX];
  __shared__ float sin_arr[SIZEX];

  // Each thread computes one sine/cosine value. For an input array of size SIZEX, sine/cosine for (SIZEX - 1) distinct exponents are used by the algorithm
  float a = (float)(i + 1);
  float p = (float)(1 + (int)floor(log2(a)));
  float m_tmp = exp2f(p);
  float tmp_tmp = a - (m_tmp/2);
  float sign = (isInverseFFT == 0) ? -1 : 1;
  float exponent = (sign) * 2 * PI * tmp_tmp / m_tmp;
  cos_arr[i] = cos(exponent);
  sin_arr[i] = sin(exponent);

  __syncthreads();

  // The outer loop that handles the logN steps of the Cooley-Tukey algorithm
  for (int m = 2; m <= size; m = m*2) {
    int tmp = i / m;
    int k = tmp * m;
    int j = i - k;
    tmp = (j % (m/2));
    int q = (m/2) + tmp - 1;
    float cos_val = cos_arr[q];
    float sin_val = sin_arr[q];

    // Compute the indices of the two elements 'u' and 't' that would be used to compute the value for the current cell.
    int index1 = (j < m/2)? i : i - (m/2);  // 'u'
    int index2 = (j >= m/2)? i : i + (m/2); // 't'

    float real_value = 0;
    float imag_value = 0;
    float a_real = real_image_buf[index1];
    float a_imag = imag_image_buf[index1];
    float b_real = real_image_buf[index2];
    float b_imag = imag_image_buf[index2];

    sign = (j < m/2) ? 1 : -1; // depending on whether the current cell is left or right of m/2, we would either compute (u+t) or (u-t) respectively.
    real_value = a_real + (sign * ((b_real * cos_val) - (b_imag * sin_val)));
    imag_value = a_imag + (sign * ((b_imag * cos_val) + (b_real * sin_val)));

    __syncthreads();

    real_image_buf[i] = real_value;
    imag_image_buf[i] = imag_value;

    __syncthreads();
  }

  // Scale down the final values if computing ifftx/iffty
  if (isInverseFFT != 0) {
    real_image_buf[i] = real_image_buf[i] / size;
    imag_image_buf[i] = imag_image_buf[i] / size;
  } 

  // Optimization: This avoids the need to call cuda_filter after this function
  //               since the values that would be filtered out written as 0
  int b = blockIdx.x;
  int eightX = size/8;
  int eightY = size/8;
  int eight7Y = size - eightY;

  int factor = ((enableFilter == 1) && !(b < eightX && i < eightY) && !(b < eightX && i >= eight7Y) && !(b >= eight7Y && i < eightY) && !(b >= eight7Y && i >= eight7Y)) ? 0 : 1;
  real_image_buf[i] *= factor;
  imag_image_buf[i] *= factor;

  real_image[i * stride] = real_image_buf[i];
  imag_image[i * stride] = imag_image_buf[i];
}

__global__ void cuda_fftx(float *real_image, float *imag_image, int size_x, int size_y)
{
  int x = blockIdx.x; // each row of the image is processed by a different thread block
  compute_fft(&real_image[x*size_x], &imag_image[x*size_x], size_x, 1, 0, 0);
}

__global__ void cuda_ffty(float *real_image, float *imag_image, int size_x, int size_y)
{
  int y = blockIdx.x; // each column is processed by a different thread block.
  compute_fft(&real_image[y], &imag_image[y], size_x, size_x, 0, 1);
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
    real_image[x*size_x + y] = 0;
    imag_image[x*size_x + y] = 0;
  }
}

__global__ void cuda_ifftx(float *real_image, float *imag_image, int size_x, int size_y)
{
  int x = blockIdx.x; // each row of the image is processed by a different thread block
  compute_fft(&real_image[x*size_x], &imag_image[x*size_x], size_x, 1, 1, 0);
}

__global__ void cuda_iffty(float *real_image, float *imag_image, int size_x, int size_y)
{
  int y = blockIdx.x; // each column is processed by a different thread block.
  compute_fft(&real_image[y], &imag_image[y], size_x, size_x, 1, 0);
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
  //cuda_filter<<<numBlocks,numThreadsPerBlock,0,filterStream>>>(device_real,device_imag,size_x,size_y);
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

