#include "gpufastq/codec_gpu_nvcomp.cuh"

#include <cuda_runtime.h>
#include <nvcomp.h>
#include <nvcomp/zstd.hpp>

#include <stdexcept>
#include <string>
#include <vector>

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

void check_nvcomp_status(const nvcompStatus_t *status_ptr,
                         const char *operation) {
  if (status_ptr == nullptr) {
    throw std::runtime_error(std::string("nvcomp returned a null status for ") +
                             operation);
  }
  if (*status_ptr != nvcompSuccess) {
    throw std::runtime_error(std::string("nvcomp failed during ") + operation +
                             " with status " +
                             std::to_string(static_cast<int>(*status_ptr)));
  }
}

nvcomp::ZstdManager make_manager(size_t chunk_size, cudaStream_t stream) {
  return nvcomp::ZstdManager(
      chunk_size, nvcompBatchedZstdCompressDefaultOpts,
      nvcompBatchedZstdDecompressDefaultOpts, stream, nvcomp::NoComputeNoVerify,
      nvcomp::BitstreamKind::NVCOMP_NATIVE);
}

} // namespace

ZstdCompressedBlock nvcomp_zstd_compress(const std::vector<uint8_t> &input,
                                         size_t chunk_size,
                                         cudaStream_t stream) {
  if (input.empty()) {
    return {};
  }

  uint8_t *d_input = nullptr;
  try {
    CUDA_CHECK(cudaMalloc(&d_input, input.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_input, input.data(), input.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto result =
        nvcomp_zstd_compress_device(d_input, input.size(), chunk_size, stream);
    cudaFree(d_input);
    return result;
  } catch (...) {
    if (d_input != nullptr) {
      cudaFree(d_input);
    }
    throw;
  }
}

ZstdCompressedBlock nvcomp_zstd_compress_device(const uint8_t *d_input,
                                                size_t input_size,
                                                size_t chunk_size,
                                                cudaStream_t stream) {
  ZstdCompressedBlock result;
  result.original_size = input_size;

  if (input_size == 0) {
    return result;
  }

  auto manager = make_manager(chunk_size, stream);
  auto comp_config = manager.configure_compression(input_size);

  uint8_t *d_output = nullptr;
  size_t *d_comp_size = nullptr;

  try {
    CUDA_CHECK(cudaMalloc(&d_output, comp_config.max_compressed_buffer_size));
    CUDA_CHECK(cudaMalloc(&d_comp_size, sizeof(size_t)));

    manager.compress(d_input, d_output, comp_config, d_comp_size);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_nvcomp_status(comp_config.get_status(), "compression");

    size_t compressed_size = 0;
    CUDA_CHECK(cudaMemcpy(&compressed_size, d_comp_size, sizeof(size_t),
                          cudaMemcpyDeviceToHost));

    result.payload.resize(compressed_size);
    CUDA_CHECK(cudaMemcpy(result.payload.data(), d_output, compressed_size,
                          cudaMemcpyDeviceToHost));
  } catch (...) {
    if (d_output != nullptr) {
      cudaFree(d_output);
    }
    if (d_comp_size != nullptr) {
      cudaFree(d_comp_size);
    }
    throw;
  }

  cudaFree(d_output);
  cudaFree(d_comp_size);
  return result;
}

void nvcomp_zstd_decompress_to_device(const ZstdCompressedBlock &compressed,
                                      uint8_t **d_output, size_t *output_size,
                                      size_t chunk_size,
                                      cudaStream_t stream) {
  if (d_output == nullptr || output_size == nullptr) {
    throw std::runtime_error("Output pointers for decompression are null");
  }

  *d_output = nullptr;
  *output_size = 0;

  if (compressed.payload.empty() || compressed.original_size == 0) {
    return;
  }

  auto manager = make_manager(chunk_size, stream);

  uint8_t *d_comp = nullptr;
  try {
    CUDA_CHECK(cudaMalloc(&d_comp, compressed.payload.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_comp, compressed.payload.data(),
                               compressed.payload.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto decomp_config = manager.configure_decompression(d_comp);
    check_nvcomp_status(decomp_config.get_status(), "decompression configure");

    *output_size = decomp_config.decomp_data_size;
    CUDA_CHECK(cudaMalloc(d_output, *output_size));

    uint8_t *decomp_buffers[] = {*d_output};
    const uint8_t *comp_buffers[] = {d_comp};
    std::vector<nvcomp::DecompressionConfig> configs{decomp_config};

    manager.decompress(decomp_buffers, comp_buffers, configs);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check_nvcomp_status(configs[0].get_status(), "decompression");
  } catch (...) {
    if (*d_output != nullptr) {
      cudaFree(*d_output);
      *d_output = nullptr;
    }
    if (d_comp != nullptr) {
      cudaFree(d_comp);
    }
    *output_size = 0;
    throw;
  }

  cudaFree(d_comp);
}

std::vector<uint8_t> nvcomp_zstd_decompress(const ZstdCompressedBlock &compressed,
                                            size_t chunk_size,
                                            cudaStream_t stream) {
  if (compressed.payload.empty() || compressed.original_size == 0) {
    return {};
  }

  uint8_t *d_output = nullptr;
  size_t output_size = 0;

  try {
    nvcomp_zstd_decompress_to_device(compressed, &d_output, &output_size,
                                     chunk_size, stream);

    std::vector<uint8_t> output(output_size);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size,
                          cudaMemcpyDeviceToHost));
    cudaFree(d_output);
    return output;
  } catch (...) {
    if (d_output != nullptr) {
      cudaFree(d_output);
    }
    throw;
  }
}

} // namespace gpufastq
