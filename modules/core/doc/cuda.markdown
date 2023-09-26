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

## Getting Started

### Moving from cv to cv::cuda

Once you have built OpenCV with the CUDA modules moving to CUDA is trivial and just envolves switching to the cv::cuda namespace and using cv::cuda::GpuMat instead of cv::Mat as your main container class.  That said getting the most performance from CUDA can be more envolved depending on your work flow.  The next example quickly demostrates the steps required to exactly mirror a host side resize operation on the device - poor performance.  After discussing some of the finer points of the OpenCV CUDA API we will revisit this example to discuss its performance.


Where the objective is to perform the same
#### Example
This example shows how to perform a host-side resize operation on the device using CUDA.  That is the objective here is to resize a `src` cv::Mat to a `dst` cv::Mat using the device. Starting with the host side operation show below.
```
Mat src(1080, 1920, CV_8UC3), dst(720, 1280, CV_8UC3);
cv::resize(src, dst, dst.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
```
We can obtain the same result, a resized `dst` cv::Mat, by performing the following three steps:
1. Upload the `src` cv::Mat to a cv::cuda::GpuMat.
2. Call cv::cuda::resize, the function in the cuda namespace corresponding to cv::cuda::resize.
3. Download the result back to the `dst` cv::Mat.

```
GpuMat srcDevice(src);
cv::cuda::resize(srcDevice, dstDevice, dst.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
dstDevice.download(dst);
```




E.g. Directly converting a host side resize operation

below host-side resize operation on the device using CUDA


Resizeing `src` to `dst`

Preforming the below which resizes `src` to `dst` using cubic interpolation on the device instead of the host
```
Mat src(1080, 1920, CV_8UC3), dst(720, 1280, CV_8UC3);
cv::resize(src, dst, dst.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
```

can be achieved on the device by simply uploading src to a GpuMat calling the function in the cuda namespace (cv::cuda::resize) corresponding to the cv::resize function and then downloding the result back to the host.


on the device only requires two changes.
 1. Transfering the `src` cv::Mat to the device by uploading it to a cv::cuda::GpuMat and
 2. Calling the device version of cv::resize (cv::cuda::resize) corresponding to the cv::resize function and then downloding the result back to the host.
```
GpuMat srcDevice(src), dstDevice(dst.size(), dst.type());
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
dstDevice.download(dst);
```

@warning The example above shows how to use the cv::cuda namespace and the GpuMat class, but it is not optimal for performance. A more realistic scenario for using CUDA is when you have multiple complex operations to perform on the device. As explained in the next section, there are several factors that affect the speed and efficiency of CUDA operations, such as data transfer, memory allocation, and kernel launch. Uploading and downloading data for each operation is very costly and should be avoided as much as possible. Moreover, some OpenCV functions have subtle differences when used in the cuda namespace, which may also impact the performance. Therefore, you should always measure and compare the CPU and GPU performance for your specific use case and optimize your code accordingly.

### Differences between host and device functions
Although the cv::cuda namespace provides many functions that are similar to the host functions in the cv namespace, there are some important differences that you should be aware of when using the CUDA API. These differences are mainly related to the availability, functionality, and performance of the device functions.

 - **Availability**: Not all host functions have a corresponding device implementation. You can check the [OpenCV documentation] for the list of supported device functions and their signatures. Some device functions may have slightly different signatures than their host counterparts, depending on whether all of the functionality was implemented in CUDA or not. For example, cv::remap supports cv:: CV_16SC2, CV_32FC1, and CV_32FC2 mapping arrays, whereas cv::cuda::remap only supports CV_32FC1.

 - **Functionality**: Some device functions may not support all of the container types and/or parameter values that are supported by the host functions. For example, cv::cuda::resize does not support InterpolationFlags::INTER_AREA interpolation method, whereas cv::resize does. You should always check the documentation and the source code of the device functions to see what they can and cannot do.

 - **Performance**: The base storage class for CUDA is cv::cuda::GpuMat, whose memory is allocated on the GPU. This means that any functions with an cv::InputArray or cv::OutputArray argument require a cuda::GpuMat. A cuda::GpuMat can be constructed in many ways (e.g., dstDevice(dst.size(), dst.type())), but it always involves allocation of memory on the device. This operation has a significant overhead and for maximum performance should be performed during initialization, either explicitly or using cv::cuda::BufferPool. Additionally, when transferring data between the host and device, either by constructing a cuda::GpuMat from a cv::Mat (e.g., GpuMat srcDevice(src)) or explicitly uploading/downloading (e.g., dstDevice.download(dst)) to/from the device, you should be aware of the bandwidth and latency costs of data transfer. You should minimize the number and size of data transfers as much as possible and use asynchronous methods when available.

### Synchronous and streamed modes of operation

Quick summary and then subsections:
OpenCV CUDA modules have two different modes of operation: synchronous and streamed.  To fully understand the implications of the streamed API it is recommended to first understand how CUDA streams operate be refering to the Nvidia documentation before reading the below to understand how they are used in the context of OpenCV.  That said the important distinction between the two can be summarized as:
 - **Synchronous**: functions are synchronous with respect to the host.
 - **Streamed**: where possible functions are asynchronous with respect to the host with device operations placed in CUDA streams (see cv::cuda::Stream for more details).

 First a quick detore high level overview of CUDA job submission
  - latency - call function job submitted delay then execution
  - when an OpenCV CUDA function is called which performs some work on the device, under the hood a "job" is submitted? runtime api call, check nsight compute - can show from example.  This could be after brief descritpion of operation

Description of sunch async operation

Brief detore

Implications

#### Synchronous Calls

Any CUDA function where you omit the cv::cuda::Stream parameter or you pass cv::cuda::Stream::Null() will be sychronous with respect to the host.  That is the following two function calls are equivelent and when control returns to the host in both cases `dstDevice` contains the result of the resize operation on the device.
```
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC, cuda::Stream::Null());
```
Comment on device synchronize..

This has two implications:
1. The host will be stalled performing no useful work until the resize operation returns. Also cannot submit any other jobs to the GPU, see .
2.


#### Streamed Calls

Any CUDA function which you pass a cuda::Stream to which is not null (cv::cuda::Stream::Null()) will where possible be asynchronous and have its device operations performed in that CUDA stream. e.g.
```
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC, cuda::Stream::Null());
```


```
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC, stream);
```
with all other functions being synchronous



OpenCV CUDA modules have two different modes of operation: synchronous and streamed.  To fully understand the implications of the streamed API it is recommended to first understand how CUDA streams operate be refering to the Nvidia documentation before reading the below to understand how they are used in the context of OpenCV.  That said the important distinction between the two can be summarized as:
 - **Synchronous**: functions are synchronous with respect to the host. e.g. when the following returns control to the host `dstDevice` contains the resized version of `srcDevice`.
 ```
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
 ```
 This means that the host thread cannot perform any operations
 - **Streamed**: where possible utilizes CUDA stream API (see cv::cuda::Stream for more details) and places device operations in a CUDA stream. E.g. When the following returns control to the host the contents of `dstDevice` is not guaranteed to contain the result of the resize operation.
 ```
cv::cuda::Stream stream;
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC, stream);
```
In this scenario to guarnatee this `stream.waitForCompletion` would need to be called.

To request streaming

The conditions and implications for whether OpenCV uses the synchronous or streaming API are:
 - Any CUDA function which you pass a cuda::Stream to which is not null (cv::cuda::Stream::Null()) will where possible have their device operations performed in that CUDA stream. e.g.
```
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC, stream);
```
all other functions will be synchronous e.g. both the following are equivalent to each other
```
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC, cuda::Stream::Null());
```

- All synchronous functions explicitly call `cudaDeviceSynchronize()` internally before exiting, stalling the device but guaranteeing that the result on the device is available when the function returns.

 - Any CUDA function which you pass a cuda::Stream to which is not null (cv::cuda::Stream::Null()) will where possible have its device operations performed in that CUDA stream.  E.g. When the cv::cuda::resize function returns in the following the contents of `dstDevice` is not guaranteed to contain the result of the resize operation.
```
cv::cuda::Stream stream;
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC, stream);
```
You need would need to call stream.waitForCompletion or perform the operation
 - All synchronous functions explicitly call `cudaDeviceSynchronize()` internally before exiting, stalling the device but guaranteeing that the result on the device is available when the function returns. For example, passing the null stream to cv::cuda::resize above makes it equivelent to the synchronous version, show code for both.
E.g. When the cv::cuda::resize function returns in the following two cases `dstDevice` is guaranteed to contain the result of the resize operation
```
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC);
cv::cuda::resize(srcDevice, dstDevice, dstDevice.size(), 0, 0, InterpolationFlags::INTER_CUBIC, cuda::Stream::Null());
```





 In all other cases the function will be synchronous with respect to the host.




OpenCV CUDA modules offer two types of APIs: synchronous and streaming.  There operation is explained below.
 - Any CUDA function
 - Any CUDA function with a cuda::Stream parameter is part of the streaming API. However, if you pass in the null stream (cv::cuda::Stream::Null()), it will behave as if you are calling its synchronous version.
 - All synchronous functions explicitly call `cudaDeviceSynchronize()` internally before exiting, stalling the device but guaranteeing that the result on the device is available when the function returns. For example, passing the null stream to cv::cuda::resize above makes it equivelent to the synchronous version, show code for both.
 - As with standard CUDA, you need to pin the host memory using cv::cuda::HostMem or cv::cuda::registerPageLocked for upload/download operations to be asynchronous with respect to the host. For example, constructing a cuda::GpuMat from a non-pinned cv::Mat is not asynchronous.
 - Some OpenCV CUDA functions which are part of the async API are not

 In the same way as other host level API wrappers around the CUDA SDK, some OpenCV CUDA functions

 Unlike standard CUDA kernel execution, which is always asynchronous with respect to the host, some OpenCV CUDA functions that are part of the async API are not. These functions either rely on intermediate results being transferred to and from the device or return results directly to the host and have internal synchronization as a result. For example, examples of these functions cv::cuda::cvtColor is not fully asynchronous even if you pass a non-null stream.





### Differences
Now we have described the similarities its time to dig in to a few of the subtle differences which you should be aware of when deciding to switch to the CUDA API

 1. Not all host functions have a corresponding device implementation and those that do may have slightly different function signatures depending on whether all of the funcitonality was implemented in CUDA or not (need reference function). Additionaly the CUDA variant may not support all of the container types and/or parameter values.  e.g. cv::remap supports cv:: CV_16SC2, CV_32FC1, and CV_32FC2 mapping arrays whereas cv::cuda::remap only supports CV_32FC1.

 2. The base storage class for CUDA is cv::cuda::GpuMat whose memory is allocated on the GPU.  This means that any functions with an cv::InputArray or cv::OutputArray argument require a cuda::GpuMat.  A cuda::GpuMat can be constructed in many ways (`dstDevice(dst.size(), dst.type())` etc.) however it always envolves allocation of memory on the device.  This operation has a significant overhead and for maximum performance should be performed during initialization either explicitly or is not possible then using cv::cuda::BufferPool.  Additionaly when transfering data between the host and device, either by constructing a cuda::GpuMat from a cv::Mat (`GpuMat srcDevice(src)`) or explitely uploading/downloading (`dstDevice.download(dst)`) to/from the device. <- what am I trying to say here?

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

OpenCV CUDA modules offer two api's, synchronous and asynchronous, these ideas will be familiar to users of CUDA however there are a number of subtle differences between the way these operations work when compared async kernel launches and sync/async memory operations when using the CUDA SDK as explained below.  The same as NPP or other libraries.

 - Any CUDA function with a Stream is part of the async api.  That said if you pass in the null stream it will behave as if you are calling its synchronous version.
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














- API Reference: functions and classes

    - @subpage cuda

      Core G-API operations - arithmetic, boolean, and other matrix
      operations;

    - @subpage cudacodec

      Core G-API operations - arithmetic, boolean, and other matrix
      operations;