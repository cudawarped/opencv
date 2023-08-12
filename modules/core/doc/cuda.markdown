# Introduction CUDA {#cuda_intro}

@tableofcontents

## General Information

The OpenCV CUDA modules are a set of classes and functions which utilize CUDA computational capabilities.
They are implemented using NVIDIA CUDA Runtime API and only support NVIDIA GPUs. The OpenCV CUDA
modules includes utility functions, low-level vision primitives, and high-level algorithms. The
utility functions and low-level primitives provide a powerful infrastructure for developing fast
vision algorithms taking advantage of CUDA whereas the high-level functionality includes some
state-of-the-art algorithms (such as stereo correspondence, face and people detectors, and others)
ready to be used by the application developers.

The CUDA modules are designed as a host-level API. This means that if you have pre-compiled OpenCV
CUDA binaries, you are not required to write any extra code to make use of the CUDA.  You will however need to install the CUDA Toolkit or distribute the required CUDA shared libaries to any machine you want to execute code compiled against those pre-compiled OpenCV CUDA binaries on.

The OpenCV CUDA modules are designed for ease of use and do not require any knowledge of CUDA.
Though, such a knowledge will certainly be useful to handle non-trivial cases or achieve the highest
performance. It is helpful to understand the cost of various operations, what the GPU does, what the
preferred data formats are, and so on. The CUDA module is an effective instrument for quick
implementation of CUDA-accelerated computer vision algorithms. However, if your algorithm involves
many simple operations, then, for the best possible performance, you may still need to write your
own kernels to avoid extra write and read operations on the intermediate results.

To enable CUDA support, install the Nvidia CUDA Toolkit and configure OpenCV using CMake with **WITH_CUDA=ON**. When the flag is set and if CUDA is installed, the full-featured OpenCV CUDA module is built. If the CUDA Toolkit is not installed, the module is still
built but at runtime all functions from the module throw Exception with cv::Error::GpuNotSupported error
code, except for cv::cuda::getCudaEnabledDeviceCount(). The latter function returns zero GPU count in
this case. Building OpenCV without CUDA support does not perform device code compilation, so it does
not require the CUDA Toolkit to be installed. Therefore, using the cv::cuda::getCudaEnabledDeviceCount()
function, you can implement a high-level algorithm that will detect GPU presence at runtime and
choose an appropriate implementation (CPU or GPU) accordingly.

## Compilation for Different NVIDIA Platforms

The NVIDIA compiler enables generating binary code (cubin and fatbin) and intermediate code (PTX).
Binary code often implies a specific GPU architecture and generation, so the compatibility with
other GPUs is not guaranteed. PTX is targeted for a virtual platform that is defined entirely by the
set of capabilities or features. Depending on the selected virtual platform, some of the
instructions are emulated or disabled, even if the real hardware supports all the features.

On the first call to an OpenCV CUDA function, if there is no compatible binary code but the target GPU has a compute capability (CC)greater than the PTX code, that code is Just-in-Time (JIT) compiled to binary code for the particular GPU by the device driver. When the target GPU has a CC lower than the PTX code, JIT fails. By
default, the OpenCV CUDA module includes:
 - Binaries for all CC's suported by the installed CUDA Toolkit (controlled by **CUDA_ARCH_BIN** in CMake)
 - PTX code for the highest supported CC (controlled by **CUDA_ARCH_PTX** in CMake)

if neither **CUDA_ARCH_BIN** or **CUDA_ARCH_PTX** are specified.  For an more details on this default behaviour see the example below.

### Example using Nvidia CUDA Toolkit 11.0
With the CUDA Tookit setting only **WITH_CUDA=ON** (i.e. leaving **CUDA_ARCH_BIN** and **CUDA_ARCH_PTX** unset), the compiler will generate binaries for all compute capabilities (CCs) from 3.5 to 8.6 that are supported by the OS. This means that OpenCV CUDA modules can run on any GPU that supports these CCs without any further configuration or penalty.  However, for newer GPU's with CC higher than 8.6 (e.g., 9.0), the automatically generated PTX code for CC 8.6 will be JIT compiled to binary code supporting that GPU the first time an OpenCV CUDA function is called, incurring a significant "one time" delay (for more details see Just-in-Time compilation in the CUDA C programming). Conversely, for older devices with CCs lower than 3.5, any OpenCV CUDA function call will throw an exception.

You can always determine at runtime whether the OpenCV GPU-built binaries (or PTX code) are
compatible with your GPU. The function cv::cuda::DeviceInfo::isCompatible() returns the compatibility
status (true/false).

## Utilizing Multiple GPUs

In the current version, each of the OpenCV CUDA algorithms can use only a single GPU. So, to utilize
multiple GPUs, you have to manually distribute the work between GPUs. Switching active device can be
done using cv::cuda::setDevice() function. For more details please read Cuda C Programming Guide.

While developing algorithms for multiple GPUs, note a data passing overhead. For primitive functions
and small images, it can be significant, which may eliminate all the advantages of having multiple
GPUs. But for high-level algorithms, consider using multi-GPU acceleration. For example, the Stereo
Block Matching algorithm has been successfully parallelized using the following algorithm:

1.  Split each image of the stereo pair into two horizontal overlapping stripes.
2.  Process each pair of stripes (from the left and right images) on a separate Fermi\* GPU.
3.  Merge the results into a single disparity map.

With this algorithm, a dual GPU gave a 180% performance increase comparing to the single Fermi GPU.
The source code for this example shown below

<details>

<summary>Stereo Multi GPU Example</summary>

@include samples/gpu/stereo_multi.cpp

</details>

can be downloaded from [here](https://github.com/opencv/opencv/tree/4.x/samples/gpu/stereo_multi.cpp).