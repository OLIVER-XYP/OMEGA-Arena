#ifndef CUDADEVICE_HPP
#define CUDADEVICE_HPP

#include <string>

namespace hlt::gpu {

struct CudaDeviceInfo {
    bool runtime_available{};
    int device_count{};
    std::string device_name;
    std::string error_message;
};

CudaDeviceInfo query_cuda_device();

} // namespace hlt::gpu

#endif // CUDADEVICE_HPP
