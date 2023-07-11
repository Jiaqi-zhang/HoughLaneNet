from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='deep_hough_transform',
    ext_modules=[
        CUDAExtension('deep_hough_transform', [
            'dht_cuda.cpp',
            'dht_cuda_kernel.cu'
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': ['-arch=sm_60']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)