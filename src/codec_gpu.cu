#include "gpufastq/codec_gpu.cuh"

#include <cub/device/device_adjacent_difference.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpufastq {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + ": " +         \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

namespace {

template <typename T> void cuda_free_if_set(T *ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

__global__ void cast_u64_to_u32_kernel(const uint64_t *input, uint32_t *output,
                                       size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const uint64_t value = input[idx];
  if (value > 0xFFFFFFFFull) {
    asm("trap;");
  }
  output[idx] = static_cast<uint32_t>(value);
}

__global__ void widen_u32_to_u64_kernel(const uint32_t *input, uint64_t *output,
                                        size_t count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  output[idx] = static_cast<uint64_t>(input[idx]);
}

__global__ void compute_quality_lengths_kernel(const uint64_t *line_offsets,
                                               uint32_t *quality_lengths,
                                               size_t num_records) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_records) {
    return;
  }

  const size_t qual_line = 4 * idx + 3;
  const uint64_t line_start = line_offsets[qual_line];
  const uint64_t next_line_start = line_offsets[qual_line + 1];
  if (next_line_start <= line_start) {
    asm("trap;");
  }

  const uint64_t length = next_line_start - line_start - 1;
  if (length > 0xFFFFFFFFull) {
    asm("trap;");
  }
  quality_lengths[idx] = static_cast<uint32_t>(length);
}

struct IsNewline {
  __host__ __device__ bool operator()(uint8_t value) const {
    return value == static_cast<uint8_t>('\n');
  }
};

} // namespace

std::vector<uint64_t>
build_line_offsets_gpu(const std::vector<uint8_t> &raw_bytes,
                       cudaStream_t stream) {
  if (raw_bytes.empty()) {
    return {0};
  }

  uint8_t *d_raw_bytes = nullptr;
  uint64_t *d_line_offsets = nullptr;

  try {
    const uint64_t file_size = static_cast<uint64_t>(raw_bytes.size());
    CUDA_CHECK(cudaMalloc(&d_raw_bytes, raw_bytes.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_raw_bytes, raw_bytes.data(), raw_bytes.size(),
                               cudaMemcpyHostToDevice, stream));

    auto d_bytes = thrust::device_pointer_cast(d_raw_bytes);
    const uint64_t internal_count = thrust::count_if(
        thrust::cuda::par.on(stream), d_bytes, d_bytes + (file_size - 1),
        IsNewline{});

    std::vector<uint64_t> line_offsets(static_cast<size_t>(internal_count + 2));
    line_offsets.front() = 0;
    line_offsets.back() = file_size;

    if (internal_count > 0) {
      CUDA_CHECK(cudaMalloc(&d_line_offsets,
                            internal_count * sizeof(uint64_t)));
      auto idx_begin = thrust::make_counting_iterator<uint64_t>(1);
      auto idx_end = thrust::make_counting_iterator<uint64_t>(file_size);
      auto d_output = thrust::device_pointer_cast(d_line_offsets);
      thrust::copy_if(thrust::cuda::par.on(stream), idx_begin, idx_end, d_bytes,
                      d_output, IsNewline{});
      CUDA_CHECK(cudaMemcpyAsync(line_offsets.data() + 1, d_line_offsets,
                                 internal_count * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cuda_free_if_set(d_line_offsets);
    cuda_free_if_set(d_raw_bytes);
    return line_offsets;
  } catch (...) {
    cuda_free_if_set(d_line_offsets);
    cuda_free_if_set(d_raw_bytes);
    throw;
  }
}

void delta_encode_offsets_to_lengths(const uint64_t *d_offsets,
                                     uint32_t *d_lengths, size_t count,
                                     cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  if (d_offsets == nullptr || d_lengths == nullptr) {
    throw std::runtime_error("Delta-encode pointers must not be null");
  }

  uint64_t *d_deltas64 = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  try {
    CUDA_CHECK(cudaMalloc(&d_deltas64, count * sizeof(uint64_t)));

    CUDA_CHECK(cub::DeviceAdjacentDifference::SubtractLeftCopy(
        nullptr, temp_storage_bytes, d_offsets, d_deltas64, count,
        cub::Difference{}, stream));
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CHECK(cub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes, d_offsets, d_deltas64, count,
        cub::Difference{}, stream));

    const uint32_t block_size = 256;
    const uint32_t grid_size =
        static_cast<uint32_t>((count + block_size - 1) / block_size);
    cast_u64_to_u32_kernel<<<grid_size, block_size, 0, stream>>>(
        d_deltas64, d_lengths, count);
    CUDA_CHECK(cudaGetLastError());
  } catch (...) {
    cuda_free_if_set(static_cast<char *>(d_temp_storage));
    cuda_free_if_set(d_deltas64);
    throw;
  }

  cuda_free_if_set(static_cast<char *>(d_temp_storage));
  cuda_free_if_set(d_deltas64);
}

void delta_decode_lengths_to_offsets(const uint32_t *d_lengths,
                                     uint64_t *d_offsets, size_t count,
                                     cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  if (d_lengths == nullptr || d_offsets == nullptr) {
    throw std::runtime_error("Delta-decode pointers must not be null");
  }

  uint64_t *d_lengths64 = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  try {
    CUDA_CHECK(cudaMalloc(&d_lengths64, count * sizeof(uint64_t)));

    const uint32_t block_size = 256;
    const uint32_t grid_size =
        static_cast<uint32_t>((count + block_size - 1) / block_size);
    widen_u32_to_u64_kernel<<<grid_size, block_size, 0, stream>>>(
        d_lengths, d_lengths64, count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                             d_lengths64, d_offsets, count,
                                             stream));
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_lengths64, d_offsets, count,
                                             stream));
  } catch (...) {
    cuda_free_if_set(static_cast<char *>(d_temp_storage));
    cuda_free_if_set(d_lengths64);
    throw;
  }

  cuda_free_if_set(static_cast<char *>(d_temp_storage));
  cuda_free_if_set(d_lengths64);
}

QualityLengthAnalysis
analyze_quality_lengths(const std::vector<uint64_t> &line_offsets,
                        uint64_t num_records, cudaStream_t stream) {
  QualityLengthAnalysis analysis;
  analysis.lengths.resize(static_cast<size_t>(num_records), 0);
  if (num_records == 0) {
    return analysis;
  }
  if (line_offsets.size() != static_cast<size_t>(4 * num_records + 1)) {
    throw std::runtime_error(
        "Line-offset count is inconsistent with FASTQ record count");
  }

  uint64_t *d_line_offsets = nullptr;
  uint32_t *d_quality_lengths = nullptr;
  uint32_t *d_min_length = nullptr;
  uint32_t *d_max_length = nullptr;
  void *d_temp_storage = nullptr;
  size_t min_temp_storage_bytes = 0;
  size_t max_temp_storage_bytes = 0;

  try {
    CUDA_CHECK(
        cudaMalloc(&d_line_offsets, line_offsets.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_quality_lengths, num_records * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_min_length, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_max_length, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_line_offsets, line_offsets.data(),
                               line_offsets.size() * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, stream));

    const uint32_t block_size = 256;
    const uint32_t grid_size =
        static_cast<uint32_t>((num_records + block_size - 1) / block_size);
    compute_quality_lengths_kernel<<<grid_size, block_size, 0, stream>>>(
        d_line_offsets, d_quality_lengths, static_cast<size_t>(num_records));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cub::DeviceReduce::Min(nullptr, min_temp_storage_bytes,
                                      d_quality_lengths, d_min_length,
                                      static_cast<size_t>(num_records), stream));
    CUDA_CHECK(cub::DeviceReduce::Max(nullptr, max_temp_storage_bytes,
                                      d_quality_lengths, d_max_length,
                                      static_cast<size_t>(num_records), stream));
    CUDA_CHECK(cudaMalloc(&d_temp_storage,
                          std::max(min_temp_storage_bytes, max_temp_storage_bytes)));
    CUDA_CHECK(cub::DeviceReduce::Min(d_temp_storage, min_temp_storage_bytes,
                                      d_quality_lengths, d_min_length,
                                      static_cast<size_t>(num_records), stream));
    CUDA_CHECK(cub::DeviceReduce::Max(d_temp_storage, max_temp_storage_bytes,
                                      d_quality_lengths, d_max_length,
                                      static_cast<size_t>(num_records), stream));

    uint32_t min_length = 0;
    uint32_t max_length = 0;
    CUDA_CHECK(cudaMemcpyAsync(analysis.lengths.data(), d_quality_lengths,
                               num_records * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&min_length, d_min_length, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&max_length, d_max_length, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    analysis.fixed_length = min_length;
    analysis.layout = (min_length == max_length)
                          ? QualityLayoutKind::FixedLength
                          : QualityLayoutKind::VariableLength;
    if (analysis.layout == QualityLayoutKind::VariableLength) {
      analysis.fixed_length = 0;
    }
  } catch (...) {
    cuda_free_if_set(static_cast<char *>(d_temp_storage));
    cuda_free_if_set(d_max_length);
    cuda_free_if_set(d_min_length);
    cuda_free_if_set(d_quality_lengths);
    cuda_free_if_set(d_line_offsets);
    throw;
  }

  cuda_free_if_set(static_cast<char *>(d_temp_storage));
  cuda_free_if_set(d_max_length);
  cuda_free_if_set(d_min_length);
  cuda_free_if_set(d_quality_lengths);
  cuda_free_if_set(d_line_offsets);
  return analysis;
}

} // namespace gpufastq
