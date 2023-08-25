# CUDA Modules Introduction {#cuda_intro}

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
CUDA binaries, you are not required to write any extra code to make use of the CUDA.  You will however need to install the CUDA Toolkit or share the CUDA libraries with any machine you use intend to run code compiled against those binaries on.

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

## API Reference: functions and classes

### Similarities
Moving to CUDA is trivial and just envolves switching to the cv::cuda namespace and using cv::cuda::GpuMat instead of cv::Mat as your main container class.  e.g. Given the below code which resize `src` to `dst` on the host using cubic interpolation
```
Mat src(1080, 1920, CV_8UC3), dst(720, 1280, CV_8UC3);
cv::resize(src, dst, dst.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
```
the same can be achieved on the device by simply uploading src to a GpuMat calling the function in the cuda namespace (cv::cuda::resize) corresponding to the cv::resize function and then downloding the result back to the host.
```
GpuMat srcDevice(src), dstDevice(dst.size(), dst.type());
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
dstDevice.download(dst);
```
@note In practice you would not do this for a single simple function like resize but the example demonstrates the principle.  Read the below.  With the above the you will find the performance on the GPU slower.  Beyond the scope of this introductory material, those familiar with CUDA will understand the consequenses of uploading and array to the device to perform a single simple operation and then downloading it directly back afterwards as being inefficient, but there are futher subtleties to be aware of when using the OpenCV api, which will cause futher performance hits with the above code.

### Differences
Now we have described the similarities its time to dig in to a few of the subtle differences which make the conversion or decision to convert less straightforward

1) Not all host functions have a corresponding device implementation and those that do may have slightly different function signatures depending on whether all of the funcitonality was implemented in CUDA or not (need reference function). Additionaly the CUDA variant may not support all of the container types and/or parameter values.  e.g. compare cv::remap with cv::cuda::remap.
2) The base storage class for CUDA is cv::cuda::GpuMat whose memory is allocated on the GPU.  This means that any functions with an cv::InputArray or cv::OutputArray argument require a cuda::GpuMat.  A cuda::GpuMat can be constructed in many ways (`dstDevice(dst.size(), dst.type())` etc.) however it always envolves allocation of memory on the device.  This operation has a significant overhead and for maximum performance should be performed at the start if possible, if this is not possible then you should consider using cv::cuda::BufferPool.  Additionaly when transfering data between the host and device, either by constructing a cuda::GpuMat from a cv::Mat (`GpuMat srcDevice(src)`) or explitely uploading/downloading (`dstDevice.download(dst)`) to/from the device.

Most pipelines have an input and output



is therefore recommended to on

OpenCV is a synchronous operation and

 has a significant overhead and for maximum performance it is recommended to avoid host device transfers as much as possible and if possible use Streams to overlap them with GPU execution.  This is the caveat mentioned above and will require you to think more carefuly about your program design, although its just a simple switch you need to be careful.



That said there are a number of important differences.  These will be discussed below with reference to the ... method which has both a .. and a ... implementation.

Or mention cv:: resize and cuda and show arguments are the same.
cv::resize and cv::cuda::resize
...

However

Not all host functions have been implemented and those that have may have slightly different function signatures depending on whether all of the funcitonality was implemented in CUDA or not. Also cuda may not support all the types.. remap different types

void cv::resize 	( 	InputArray  	src,
OutputArray  	dst,
Size  	dsize,
double  	fx = 0,
double  	fy = 0,
int  	interpolation = INTER_LINEAR
)

void cv::cuda::resize 	( 	InputArray  	src,
OutputArray  	dst,
Size  	dsize,
double  	fx = 0,
double  	fy = 0,
int  	interpolation = INTER_LINEAR,
Stream &  	stream = Stream::Null()
)

The base storage class for CUDA is GpuMat whose memory is allocated on the GPU.  This means that any functions with an InputArray or OutputArray argument require a GpuMat not a (....).  Programatically a GPUmat can be constructed from a mat with ... or .... however this has a significant overhead and for maximum performance it is recommended to avoid host device transfers as much as possible and if possible use Streams to overlap them with GPU execution.  This is the caveat mentioned above and will require you to think more carefuly about your program design, although its just a simple switch you need to be careful.

The base storage class

Any processing .  When the input or output is an array this will require a GPUMat


CUDA functions process GpuMat it both more efficient and safer to always pass in GpuMat


CUDA functions process memory stored on the GPU inside GpuMat's

CUDA functions process GpuMat's not Mat's a ... whos
upload/download

async and sync

OpenCV CUDA modules offer two api's, synchronous and asynchronous, these ideas will be familiar to users of CUDA however there are a number of subtle differences between the way these operations work when compared to sync/asyn memory operations in pure CUDA and async kernel launches as explained below.

 - Any CUDA call with a Stream is part of the async api.  That said if you pass in the null stream it will behave as if you are calling its synchronous version.
 - All synchronous versions explicitly call cudaDeviceSync before exiting stalling the device but guranteeing that the result on the device is available when the funciton returns. e.g. passing the null stream both this and .. are sync .. is not
 - As with standard CUDA you need to pin the host memory Host.. or ... for upload download to be asyncronous with respect to the host. e.g. ... is not async
 - Unlike standard CUDA kernel execution which is always async with respect to the host some OpenCV CUDA functions which are part of the async api are not.  These funcitons rely on intermediate retults being transfered to and from the device and have internal synchronization as a result.

Whilst the sync is a nice safe way to experiment due without having to remember ... always try to use
 Take home, always try to use the ... any call to .. will stall all work on the device even asynchronous calls in .....


it is important to understand the subl

When using the Python bindings it is important to pass the .. For example .... .

cv:: to cuda:: is trivu

### API Reference: functions and classes
@subpage cuda

GpuMat upload download
Sync like cv:: Async returns to host before finished, not like CUDA kernel mem async where returns imidiately as their may be intermediate sync steps, intermediate results which other kernels memory transactions rely on ....

Python bindings overview

## Compilation for Different NVIDIA Platforms

The NVIDIA compiler enables generating binary code (cubin and fatbin) and intermediate code (PTX).
Binary code often implies a specific GPU architecture and generation, so the compatibility with
other GPUs is not guaranteed. PTX is targeted for a virtual platform that is defined entirely by the
set of capabilities or features. Depending on the selected virtual platform, some of the
instructions are emulated or disabled, even if the real hardware supports all the features.

On the first call to an OpenCV CUDA function, if there is no compatible binary code but the target GPU has a compute capability (CC)greater than the PTX code, that code is Just-in-Time (JIT) compiled to binary code for the particular GPU by the device driver. When the target GPU has a CC lower than the PTX code, JIT fails. By
default, the OpenCV CUDA module includes:
 - Binaries for all CC's suported by the installed CUDA Toolkit (set using **CUDA_ARCH_BIN=X.x;Y.y;...** in CMake)
 - PTX code for the highest supported CC (set using **CUDA_ARCH_PTX=X.x;Y.y;...** in CMake)

if neither **CUDA_ARCH_BIN** or **CUDA_ARCH_PTX** are specified.  For an more details on this default behaviour see the example below.

### Example using Nvidia CUDA Toolkit 11.0
With the CUDA Tookit installed setting only **WITH_CUDA=ON** (i.e. leaving **CUDA_ARCH_BIN** and **CUDA_ARCH_PTX** unset), the compiler will generate binaries for all compute capabilities (CCs) from 3.5 to 8.6 that are supported by the OS. This means that OpenCV CUDA modules can run on any GPU that supports these CCs without any further configuration or penalty.  However, for newer GPU's with CC higher than 8.6 (e.g., 9.0), the automatically generated PTX code for CC 8.6 will be JIT compiled to binary code supporting that GPU the first time an OpenCV CUDA function is called.  This incurrs a significant "one time" delay (for more details see Just-in-Time compilation in the CUDA C programming). Conversely, for older devices with CCs lower than 3.5, any OpenCV CUDA function call will throw an exception.

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


- API Reference: functions and classes

    - @subpage cuda

      Core G-API operations - arithmetic, boolean, and other matrix
      operations;

    - @subpage cudacodec

      Core G-API operations - arithmetic, boolean, and other matrix
      operations;