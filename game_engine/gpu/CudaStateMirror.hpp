#ifndef CUDASTATEMIRROR_HPP
#define CUDASTATEMIRROR_HPP

#include <memory>

#include "StateFrame.hpp"

namespace hlt::gpu {

class CudaStateMirror {
public:
    CudaStateMirror();
    ~CudaStateMirror();

    CudaStateMirror(const CudaStateMirror &) = delete;
    CudaStateMirror &operator=(const CudaStateMirror &) = delete;

    void upload(const StateFrame &frame);
    StateFrame download() const;

    bool has_device_storage() const;
    std::size_t cell_count() const;
    std::size_t entity_count() const;
    std::size_t player_count() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace hlt::gpu

#endif // CUDASTATEMIRROR_HPP
