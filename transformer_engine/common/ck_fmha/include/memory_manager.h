#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_MEMORY_MANAGER_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_ROCM_MEMORY_MANAGER_H_

#include <hip/hip_runtime.h>

#include <map>
#include <stdexcept>
#include <vector>

#define CHECK_HIP_ERROR(expr)                                                            \
    do {                                                                                 \
        const hipError_t error_code = (expr);                                            \
        if (error_code != hipSuccess) {                                                  \
            std::string error_str = hipGetErrorString(error_code);                       \
            std::cerr << "HIP error: " << error_str << " in " << __FILE__ << " at line " \
                      << __LINE__ << std::endl;                                          \
            throw std::runtime_error(error_str);                                         \
        }                                                                                \
    } while (0)

class MemoryManager {
   public:
    MemoryManager(hipStream_t stream) : stream_(stream) {};
    ~MemoryManager() {
        for (auto ptr : device_memory_) {
            const hipError_t err = hipFreeAsync(ptr, stream_);
            if (err != hipSuccess) {
                std::cerr << "Failed to free device memory: " << hipGetErrorString(err)
                          << std::endl;
            }
        }
    }

    void* allocate(size_t size) {
        // Find a free block of memory that is large enough
        auto iter = free_blocks_.lower_bound(size);
        if (iter != free_blocks_.end()) {
            // clang-format off
            size_t block_size = iter->first;
            void* ptr = iter->second;
            // clang-format on
            free_blocks_.erase(iter);
            if (block_size > size) {
                // Split the block
                void* new_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(ptr) + size);
                free_blocks_[block_size - size] = new_ptr;
            }
            allocated_blocks_[ptr] = size;
            return ptr;
        }

        // No suitable free block found, allocate new memory
        void* ptr;
        CHECK_HIP_ERROR(hipMallocAsync(&ptr, size, stream_));
        allocated_blocks_[ptr] = size;
        device_memory_.push_back(ptr);
        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) {
            return;
        }
        auto iter = allocated_blocks_.find(ptr);
        if (iter == allocated_blocks_.end()) {
            throw std::runtime_error("deallocate: invalid pointer");
        }

        size_t size = iter->second;
        allocated_blocks_.erase(iter);
        free_blocks_[size] = ptr;
    }

   private:
    hipStream_t stream_;
    std::map<size_t, void*> free_blocks_;
    std::map<void*, size_t> allocated_blocks_;
    std::vector<void*> device_memory_;
};

#endif