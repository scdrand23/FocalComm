import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, sources):
    return CUDAExtension(
        name=name,
        sources=sources,
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '-gencode=arch=compute_90,code=sm_90'  # H100 architecture
            ]
        },
        include_dirs=[os.path.abspath('iou3d_nms/src')]
    )


setup(
    name='iou3d_nms_cuda',
    ext_modules=[
        make_cuda_ext(
                name='iou3d_nms_cuda',
                # module='pcdet.ops.iou3d_nms',
                sources=[
                    'iou3d_nms/src/iou3d_cpu.cpp',
                    'iou3d_nms/src/iou3d_nms_api.cpp',
                    'iou3d_nms/src/iou3d_nms.cpp',
                    'iou3d_nms/src/iou3d_nms_kernel.cu',
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)  # Disable ninja build
    }
)