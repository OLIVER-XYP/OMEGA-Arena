#include "CudaDevice.hpp"

#include <cuda_runtime.h>

namespace hlt::gpu {

CudaDeviceInfo query_cuda_device() {
    CudaDeviceInfo info;
    int count = 0;
    const auto count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess) {
        info.error_message = cudaGetErrorString(count_status);
        return info;
    }

    info.device_count = count;
    if (count <= 0) {
        info.error_message = "no CUDA devices found";
        return info;
    }

    cudaDeviceProp prop{};
    const auto prop_status = cudaGetDeviceProperties(&prop, 0);
    if (prop_status != cudaSuccess) {
        info.error_message = cudaGetErrorString(prop_status);
        return info;
    }

    info.runtime_available = true;
    info.device_name = prop.name;
    return info;
}

} // namespace hlt::gpu
