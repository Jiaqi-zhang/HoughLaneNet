#include <ATen/ATen.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> line_accum_cuda_forward(torch::Tensor       output,
                                                   const torch::Tensor input,
                                                   const float *       tabCos,
                                                   const float *       tabSin,
                                                   const int           numangle,
                                                   const int           numrho);

std::vector<torch::Tensor> line_accum_cuda_backward(torch::Tensor       output,
                                                    const torch::Tensor input,
                                                    const float *       tabCos,
                                                    const float *       tabSin,
                                                    const int           numangle,
                                                    const int           numrho);

// C++ interface

#define CHECK_CUDA(x)       AT_ASSERT(x.type().is_cuda())  //, #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())   //, #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define PI 3.14159265358979323846

void initTab(float *   tabSin,
             float *   tabCos,
             const int numangle,
             const int numrho,
             const int H,
             const int W)
{
    float irho   = int(std::sqrt(H * H + W * W) + 1) / float((numrho - 1));
    float itheta = PI / numangle;
    float angle  = 0;
    for (int i = 0; i < numangle; ++i) {
        tabCos[i] = std::cos(angle) / irho;
        tabSin[i] = std::sin(angle) / irho;
        angle += itheta;
    }
}

std::vector<at::Tensor>
line_accum_forward(at::Tensor output, const at::Tensor input, const int numangle, const int numrho)
{
    CHECK_INPUT(output);
    CHECK_INPUT(input);

    const int H             = input.size(2);
    const int W             = input.size(3);

    float tabSin[numangle], tabCos[numangle];
    initTab(tabSin, tabCos, numangle, numrho, H, W);

    auto out = line_accum_cuda_forward(output, input, tabCos, tabSin, numangle, numrho);
    CHECK_CONTIGUOUS(out[0]);
    return out;
}

std::vector<torch::Tensor> line_accum_backward(torch::Tensor output,
                                               torch::Tensor input,
                                               const int     numangle,
                                               const int     numrho)
{
    CHECK_INPUT(output);
    CHECK_INPUT(input);

    const int H             = output.size(2);
    const int W             = output.size(3);

    float tabSin[numangle], tabCos[numangle];
    initTab(tabSin, tabCos, numangle, numrho, H, W);

    return line_accum_cuda_backward(output,
                                    input,
                                    tabCos,
                                    tabSin,
                                    numangle,
                                    numrho);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &line_accum_forward, "line features accumulating forward (CUDA)");
    m.def("backward", &line_accum_backward, "line features accumulating backward (CUDA)");
}