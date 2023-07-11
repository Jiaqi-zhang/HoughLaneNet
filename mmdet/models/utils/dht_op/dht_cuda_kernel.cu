#include <ATen/ATen.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>

// -------
// KERNELS
// -------

__global__ void line_accum_forward_kernel(float *      output,
                                          const float *__restrict__ input,
                                          const float *tabCos,
                                          const float *tabSin,
                                          const int    imWidth,
                                          const int    imHeight,
                                          const int    threadW,
                                          const int    threadH,
                                          const int    threadK,
                                          const int    channelSize,
                                          const int    batchSize,
                                          const int    numangle,
                                          const int    numrho)
{
    int batch   = blockIdx.y;
    int channel = blockIdx.x;
    int x       = threadIdx.x * threadW;
    int y       = threadIdx.y * threadH;
    int k       = threadIdx.z * threadK;

    int imgStartIdx =
        batch * channelSize * imWidth * imHeight + channel * imWidth * imHeight + y * imWidth + x;

    int angleStartIdx = k;

    if (x < imWidth && y < imHeight && channel < channelSize && batch < batchSize && k < numangle) {
        int imgIndex = imgStartIdx;
        int angleIndex;
        int outIndex;
        int r;
        for (int idY = 0; idY < threadH; idY++) {
            imgIndex = imgStartIdx + idY * imWidth;
            if (y + idY < imHeight) {
                for (int idX = 0; idX < threadW; idX++) {
                    if (x + idX < imWidth) {
                        for (int idK = 0; idK < threadK; idK++) {
                            angleIndex = angleStartIdx + idK;
                            if (angleIndex < numangle) {
                                int xx = x + idX - imWidth / 2, yy = y + idY - imHeight / 2;
                                r = std::round(float(xx) * (tabCos[angleIndex])
                                               + float(yy) * (tabSin[angleIndex]));
                                r += ((numrho) / 2);
                                outIndex = batch * channelSize * numangle * numrho
                                           + numangle * numrho * channel + angleIndex * numrho + r;
                                float val = input[imgIndex];
                                atomicAdd(&(output[outIndex]), val);
                            }
                            else
                                break;
                        }
                        imgIndex++;
                    }
                    else
                        break;
                }
            }
            else
                break;
        }
    }
}

__global__ void line_accum_backward_kernel(float *      output,
                                           const float *__restrict__ input,
                                           const float *tabCos,
                                           const float *tabSin,
                                           const int    imWidth,
                                           const int    imHeight,
                                           const int    threadW,
                                           const int    threadH,
                                           const int    threadK,
                                           const int    channelSize,
                                           const int    batchSize,
                                           const int    numangle,
                                           const int    numrho)
{
    int batch   = blockIdx.y;
    int channel = blockIdx.x;
    int x       = threadIdx.x * threadW;
    int y       = threadIdx.y * threadH;
    int k       = threadIdx.z * threadK;

    int imgStartIdx =
        batch * channelSize * imWidth * imHeight + channel * imWidth * imHeight + y * imWidth + x;
    int angleStartIdx = k;

    if (x < imWidth && y < imHeight && channel < channelSize && batch < batchSize && k < numangle) {
        int imgIndex = imgStartIdx;
        int angleIndex;
        int outIndex;
        int r;
        for (int idY = 0; idY < threadH; idY++) {
            imgIndex = imgStartIdx + idY * imWidth;
            if (y + idY < imHeight) {
                for (int idX = 0; idX < threadW; idX++) {
                    if (x + idX < imWidth) {
                        for (int idK = 0; idK < threadK; idK++) {
                            angleIndex = angleStartIdx + idK;
                            if (angleIndex < numangle) {
                                int xx = x + idX - imWidth / 2, yy = y + idY - imHeight / 2;
                                r = std::round(float(xx) * tabCos[angleIndex]
                                               + float(yy) * tabSin[angleIndex]);
                                r += ((numrho) / 2);
                                outIndex = batch * channelSize * numangle * numrho
                                           + numangle * numrho * channel + angleIndex * numrho + r;
                                float val = input[outIndex];
                                atomicAdd(&(output[imgIndex]), val);
                            }
                            else
                                break;
                        }
                        imgIndex++;
                    }
                    else
                        break;
                }
            }
            else
                break;
        }
    }
}

// ---------
// Wrappers
// ---------

std::vector<torch::Tensor> line_accum_cuda_forward(torch::Tensor       output,
                                                   const torch::Tensor input,
                                                   const float *       tabCos,
                                                   const float *       tabSin,
                                                   const int           numangle,
                                                   const int           numrho)
{
    // -input: [N, C, H, W]
    // -tabCos: [numangle]
    // -tabSin: [numangle]
    const int batch_size    = input.size(0);
    const int channels_size = input.size(1);
    const int imH           = input.size(2);
    const int imW           = input.size(3);

    const int blockSizeX = std::min(8, imW);
    const int threadW    = ceil(imW / (float)blockSizeX);

    const int blockSizeY = std::min(8, imH);
    const int threadH    = ceil(imH / (float)blockSizeY);

    const int blockSizeZ = std::min(8, numangle);
    const int threadK    = ceil(numangle / (float)blockSizeZ);

    const dim3 blocks(channels_size, batch_size);
    const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float *d_tabCos, *d_tabSin;

    cudaMalloc((void **)&d_tabCos, sizeof(float) * numangle);
    cudaMalloc((void **)&d_tabSin, sizeof(float) * numangle);

    cudaMemcpy(d_tabCos, tabCos, sizeof(float) * numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float) * numangle, cudaMemcpyHostToDevice);

    line_accum_forward_kernel<<<blocks, threads>>>(output.data<float>(),
                                                   input.data<float>(),
                                                   d_tabCos,
                                                   d_tabSin,
                                                   imW,
                                                   imH,
                                                   threadW,
                                                   threadH,
                                                   threadK,
                                                   channels_size,
                                                   batch_size,
                                                   numangle,
                                                   numrho);

    cudaFree(d_tabCos);
    cudaFree(d_tabSin);
    return {output};
}

std::vector<torch::Tensor> line_accum_cuda_backward(torch::Tensor       output,
                                                    const torch::Tensor input,
                                                    const float *       tabCos,
                                                    const float *       tabSin,
                                                    const int           numangle,
                                                    const int           numrho)
{
    const int batch_size    = output.size(0);
    const int channels_size = output.size(1);
    const int imH           = output.size(2);
    const int imW           = output.size(3);

    const int blockSizeX = std::min(8, imW);
    const int threadW    = ceil(imW / (float)blockSizeX);

    const int blockSizeY = std::min(8, imH);
    const int threadH    = ceil(imH / (float)blockSizeY);

    const int blockSizeZ = std::min(8, numangle);
    const int threadK    = ceil(numangle / (float)blockSizeZ);

    const dim3 blocks(channels_size, batch_size);
    const dim3 threads(blockSizeX, blockSizeY, blockSizeZ);

    float *d_tabCos, *d_tabSin;

    cudaMalloc((void **)&d_tabCos, sizeof(float) * numangle);
    cudaMalloc((void **)&d_tabSin, sizeof(float) * numangle);

    cudaMemcpy(d_tabCos, tabCos, sizeof(float) * numangle, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tabSin, tabSin, sizeof(float) * numangle, cudaMemcpyHostToDevice);

    line_accum_backward_kernel<<<blocks, threads>>>(output.data<float>(),
                                                    input.data<float>(),
                                                    d_tabCos,
                                                    d_tabSin,
                                                    imW,
                                                    imH,
                                                    threadW,
                                                    threadH,
                                                    threadK,
                                                    channels_size,
                                                    batch_size,
                                                    numangle,
                                                    numrho);

    cudaFree(d_tabCos);
    cudaFree(d_tabSin);
    return {output};
}
