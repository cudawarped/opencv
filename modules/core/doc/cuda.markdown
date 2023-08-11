Introduction CUDA {#cuda_intro}
========================

@tableofcontents

General Information
-------------------

The OpenCV CUDA module is a set of classes and functions to utilize CUDA computational capabilities.
It is implemented using NVIDIA CUDA Runtime API and supports only NVIDIA GPUs. The OpenCV CUDA
module includes utility functions, low-level vision primitives, and high-level algorithms. The
utility functions and low-level primitives provide a powerful infrastructure for developing fast
vision algorithms taking advantage of CUDA whereas the high-level functionality includes some
state-of-the-art algorithms (such as stereo correspondence, face and people detectors, and others)
ready to be used by the application developers.

The CUDA module is designed as a host-level API. This means that if you have pre-compiled OpenCV
CUDA binaries, you are not required to write any extra code to make use of the CUDA.  You will however need to install the CUDA Toolkit or distribute the required CUDA shared libaries to any machine you want to execute code compiled against those pre-compiled OpenCV CUDA binaries on.

The OpenCV CUDA module is designed for ease of use and does not require any knowledge of CUDA.
Though, such a knowledge will certainly be useful to handle non-trivial cases or achieve the highest
performance. It is helpful to understand the cost of various operations, what the GPU does, what the
preferred data formats are, and so on. The CUDA module is an effective instrument for quick
implementation of CUDA-accelerated computer vision algorithms. However, if your algorithm involves
many simple operations, then, for the best possible performance, you may still need to write your
own kernels to avoid extra write and read operations on the intermediate results.

To enable CUDA support, install the Nvidia Toolkit and configure OpenCV using CMake with `WITH\_CUDA=ON`. When the flag is set and
if CUDA is installed, the full-featured OpenCV CUDA module is built. Otherwise, the module is still
built but at runtime all functions from the module throw Exception with cv::Error::GpuNotSupported error
code, except for cv::cuda::getCudaEnabledDeviceCount(). The latter function returns zero GPU count in
this case. Building OpenCV without CUDA support does not perform device code compilation, so it does
not require the CUDA Toolkit installed. Therefore, using the cv::cuda::getCudaEnabledDeviceCount()
function, you can implement a high-level algorithm that will detect GPU presence at runtime and
choose an appropriate implementation (CPU or GPU) accordingly.

Compilation for Different NVIDIA Platforms
--------------------------------------------

NVIDIA compiler enables generating binary code (cubin and fatbin) and intermediate code (PTX).
Binary code often implies a specific GPU architecture and generation, so the compatibility with
other GPUs is not guaranteed. PTX is targeted for a virtual platform that is defined entirely by the
set of capabilities or features. Depending on the selected virtual platform, some of the
instructions are emulated or disabled, even if the real hardware supports all the features.

On the first call to a CUDA function, the PTX code is Just-in-Time (JIT) compiled to binary code for the particular GPU by the device driver. When the target GPU has a compute capability (CC) lower than the PTX code, JIT fails. By
default, the OpenCV CUDA module includes:

\*
   Binaries for all compute capabilities suported by the installed CUDA Toolkit (controlled by CUDA\_ARCH\_BIN in CMake)

\*
   PTX code for the highest supported compute capability (controlled by CUDA\_ARCH\_PTX in CMake)

if neither CUDA\_ARCH\_BIN or CUDA\_ARCH\_PTX are specified.

Examples:
If Nvidia Tookit 11.0 is installed which supports compute capabilities 3.5-8.6 then OpenCV CUDA routines will be able to run on
all GPU's supporting this compute capability out of the box.  For all newer platforms (e.g. 9.0) the automatically generated PTX code for 8.6 will be JIT compiled to binary code the first time a CUDA runtime function is called, ... a significant "one time" delay (for more details see Just-in-Time compilation in the CUDA C programming). For devices with CC < 3.5 any called CUDA functions will throw and exception.

This means that for devices with CC 1.3 and 2.0 binary images are ready to run. For all newer
platforms, the PTX code for 1.3 is JIT'ed to a binary image. For devices with CC 1.1 and 1.2, the
PTX for 1.1 is JIT'ed. For devices with CC 1.0, no code is available and the functions throw
Exception. For platforms where JIT compilation is performed first, the run is slow.

On a GPU with CC 1.0, you can still compile the CUDA module and most of the functions will run
flawlessly. To achieve this, add "1.0" to the list of binaries, for example,
CUDA\_ARCH\_BIN="1.0 1.3 2.0" . The functions that cannot be run on CC 1.0 GPUs throw an exception.

You can always determine at runtime whether the OpenCV GPU-built binaries (or PTX code) are
compatible with your GPU. The function cv::cuda::DeviceInfo::isCompatible() returns the compatibility
status (true/false).

Utilizing Multiple GPUs
-----------------------

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
For a source code example, see <https://github.com/opencv/opencv/tree/4.x/samples/gpu/>.

\ref stereo_multi

\ref stereo_multi.cpp

\ref samples/gpu/video_reader.cpp


video_reader.cpp

[link](samples/gpu/video_reader.cpp)

Download the source code from
[here](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/cpp/tutorial_code/core/AddingImages/AddingImages.cpp).
@include cpp/tutorial_code/core/AddingImages/AddingImages.cpp
